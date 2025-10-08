#!/usr/bin/env python3
"""
Complete VLA Caption Generator for π0/π0.5 Models
Generates hierarchical, spatially-aware captions with object locations
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Check for CUDA availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

@dataclass
class VLACaption:
    """Caption format for VLA models"""
    task_command: str              # High-level: "clean the kitchen"
    subtask: str                   # Mid-level: "pick up the plate"
    action_description: str        # Low-level: "move gripper to plate at (0.3, 0.5)"
    objects: List[Dict[str, Any]]  # Objects with bboxes
    spatial_relations: List[str]   # "plate on counter"
    scene_type: str                # "kitchen"
    confidence: float              # Overall confidence
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_training_format(self) -> Dict[str, Any]:
        """Format for VLA model training"""
        return {
            "instruction": self.task_command,
            "subtask": self.subtask,
            "action": self.action_description,
            "objects": self.objects,
            "spatial": self.spatial_relations,
            "scene": self.scene_type,
            "confidence": self.confidence
        }
    
    def to_hierarchical_format(self) -> Dict[str, Any]:
        """π0.5-style hierarchical format"""
        return {
            "high_level": self.task_command,
            "semantic": self.subtask,
            "low_level": self.action_description,
            "grounding": {
                obj["label"]: obj["bbox"] for obj in self.objects
            }
        }


class VLACaptionGenerator:
    """Complete caption generator for Vision-Language-Action models"""
    
    def __init__(self):
        """Initialize with minimal dependencies"""
        self._load_models()
    
    def _load_models(self):
        """Load required models"""
        from transformers import (
            Blip2Processor, 
            Blip2ForConditionalGeneration,
            DetrImageProcessor, 
            DetrForObjectDetection
        )
        
        print("Loading caption model...")
        # Use smaller BLIP-2 variant for better performance
        self.caption_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            device_map="auto" if DEVICE == 'cuda' else None
        )
        if DEVICE == 'cpu':
            self.caption_model = self.caption_model.to(DEVICE)
        
        print("Loading object detection model...")
        self.detector_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.detector_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        ).to(DEVICE)
        
        print("Models loaded successfully!")
    
    def generate_caption(self, image_path: str, task_context: Optional[str] = None) -> VLACaption:
        """
        Main entry point - generates complete VLA caption
        
        Args:
            image_path: Path to image file
            task_context: Optional context like "kitchen cleaning"
        
        Returns:
            VLACaption object with all information
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect objects
        objects = self._detect_objects(image)
        
        # Generate hierarchical descriptions
        task_command = self._generate_task_command(image, task_context)
        subtask = self._generate_subtask(image, objects, task_command)
        action_desc = self._generate_action(image, objects, subtask)
        
        # Extract spatial information
        spatial_relations = self._extract_spatial_relations(objects)
        scene_type = self._identify_scene(image)
        
        # Calculate confidence
        confidence = np.mean([obj["confidence"] for obj in objects[:3]]) if objects else 0.5
        
        return VLACaption(
            task_command=task_command,
            subtask=subtask,
            action_description=action_desc,
            objects=objects,
            spatial_relations=spatial_relations,
            scene_type=scene_type,
            confidence=float(confidence)
        )
    
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects and their locations"""
        inputs = self.detector_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
        results = self.detector_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]
        
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            bbox = box.cpu().numpy().tolist()
            
            # Normalize to [0,1]
            norm_bbox = [
                bbox[0] / image.width,
                bbox[1] / image.height,
                bbox[2] / image.width,
                bbox[3] / image.height
            ]
            
            obj = {
                "label": self.detector_model.config.id2label[label.item()],
                "bbox": norm_bbox,
                "center": [
                    (norm_bbox[0] + norm_bbox[2]) / 2,
                    (norm_bbox[1] + norm_bbox[3]) / 2
                ],
                "confidence": float(score.item()),
                "size": self._get_size(norm_bbox)
            }
            objects.append(obj)
        
        # Sort by confidence, limit to top 10
        objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)[:10]
        return objects
    
    def _get_size(self, bbox: List[float]) -> str:
        """Estimate object size from bbox"""
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > 0.2:
            return "large"
        elif area > 0.05:
            return "medium"
        else:
            return "small"
    
    def _generate_task_command(self, image: Image.Image, context: Optional[str]) -> str:
        """Generate high-level task"""
        if context:
            prompt = f"This is a {context} task. What should be done? Give one short command:"
        else:
            prompt = "What task should be performed here? Give one short command:"
        
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_length=15, min_length=2)
        
        response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        
        # Clean response
        response = response.strip()
        if len(response) > 50 or not response:
            response = "organize the scene"
        
        return response
    
    def _generate_subtask(self, image: Image.Image, objects: List[Dict], task: str) -> str:
        """Generate current subtask"""
        if not objects:
            return "explore the scene"
        
        # Use detected objects in prompt
        obj_list = ", ".join([obj["label"] for obj in objects[:3]])
        prompt = f"Task: {task}. Objects: {obj_list}. Current subtask (3-5 words):"
        
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_length=20, min_length=2)
        
        response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        
        # Fallback to simple action
        if not response or len(response.split()) > 6:
            response = f"pick up {objects[0]['label']}"
        
        return response.strip()
    
    def _generate_action(self, image: Image.Image, objects: List[Dict], subtask: str) -> str:
        """Generate low-level action with position"""
        if not objects:
            return "scan environment for objects"
        
        # Find relevant object
        target_obj = None
        for obj in objects:
            if obj["label"].lower() in subtask.lower():
                target_obj = obj
                break
        
        if not target_obj:
            target_obj = objects[0]
        
        # Generate action with position
        x, y = target_obj["center"]
        action = f"move gripper to {target_obj['label']} at position ({x:.2f}, {y:.2f})"
        
        # Add size-based modifier
        if target_obj["size"] == "small":
            action += " with precision grip"
        elif target_obj["size"] == "large":
            action += " with wide grip"
        
        return action
    
    def _extract_spatial_relations(self, objects: List[Dict]) -> List[str]:
        """Extract spatial relationships"""
        relations = []
        
        # Compare pairs of objects
        for i, obj1 in enumerate(objects[:4]):
            for obj2 in objects[i+1:5]:
                rel = self._compute_relation(obj1, obj2)
                if rel:
                    relations.append(rel)
        
        return relations[:5]  # Limit to 5 relations
    
    def _compute_relation(self, obj1: Dict, obj2: Dict) -> Optional[str]:
        """Compute spatial relation between two objects"""
        c1, c2 = obj1["center"], obj2["center"]
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        
        # Vertical relations
        if abs(dx) < 0.1:
            if dy > 0.1:
                return f"{obj1['label']} above {obj2['label']}"
            elif dy < -0.1:
                return f"{obj1['label']} below {obj2['label']}"
        
        # Horizontal relations  
        if abs(dy) < 0.1:
            if dx > 0.1:
                return f"{obj1['label']} left of {obj2['label']}"
            elif dx < -0.1:
                return f"{obj1['label']} right of {obj2['label']}"
        
        # Proximity
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.15:
            return f"{obj1['label']} near {obj2['label']}"
        
        return None
    
    def _identify_scene(self, image: Image.Image) -> str:
        """Identify scene type"""
        prompt = "What room or scene is this? Answer in one word:"
        
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_length=10, min_length=1)
        
        scene = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        scene = scene.strip().lower()
        
        # Map to standard types
        scene_map = {
            "kitchen": ["kitchen", "cooking", "dining"],
            "bedroom": ["bedroom", "bed", "sleeping"],
            "living_room": ["living", "lounge", "sofa"],
            "office": ["office", "desk", "workspace"],
            "bathroom": ["bathroom", "bath", "toilet"]
        }
        
        for standard, keywords in scene_map.items():
            if any(kw in scene for kw in keywords):
                return standard
        
        return scene if scene else "general"
    
    def process_batch(self, image_paths: List[str], output_file: str = "captions.json"):
        """Process multiple images and save results"""
        results = []
        
        for i, path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {path}")
            try:
                caption = self.generate_caption(path)
                result = caption.to_training_format()
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {len(results)} captions to {output_file}")
        return results


def main():
    """Example usage and test"""
    import sys
    
    # Initialize generator
    print("Initializing VLA Caption Generator...")
    generator = VLACaptionGenerator()
    
    # Process command line arguments or use example
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        task_context = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Use example image
        import urllib.request
        import tempfile
        
        print("\nDownloading example image...")
        url = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800"
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            image_path = tmp.name
        task_context = "kitchen cleaning"
        print(f"Using example image: {image_path}")
    
    # Generate caption
    print(f"\nGenerating caption for: {image_path}")
    if task_context:
        print(f"Task context: {task_context}")
    
    caption = generator.generate_caption(image_path, task_context)
    
    # Display results
    print("\n" + "="*60)
    print("VLA CAPTION RESULTS")
    print("="*60)
    
    print(f"\n Task Command: {caption.task_command}")
    print(f"Subtask: {caption.subtask}")
    print(f"Action: {caption.action_description}")
    print(f"Scene: {caption.scene_type}")
    print(f"Confidence: {caption.confidence:.2%}")
    
    print(f"\n Objects Detected ({len(caption.objects)}):")
    for obj in caption.objects[:5]:
        print(f"  • {obj['label']}: pos({obj['center'][0]:.2f}, {obj['center'][1]:.2f}), "
              f"size={obj['size']}, conf={obj['confidence']:.2%}")
    
    if caption.spatial_relations:
        print(f"\n Spatial Relations:")
        for rel in caption.spatial_relations:
            print(f"  • {rel}")
    
    # Show training formats
    print("\n" + "="*60)
    print("TRAINING FORMATS")
    print("="*60)
    
    print("\n1. Standard Training Format:")
    print(json.dumps(caption.to_training_format(), indent=2))
    
    print("\n2. Hierarchical Format (π0.5 style):")
    print(json.dumps(caption.to_hierarchical_format(), indent=2))
    
    print("\n✅ Caption generation complete!")


if __name__ == "__main__":
    main()