#!/usr/bin/env python3
"""
Complete VLA Caption Generator for π0/π0.5 Models
Generates hierarchical, spatially-aware captions with object locations
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
            dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
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
            ids = self.caption_model.generate(**inputs, max_new_tokens=15, min_new_tokens=2)
        
        # The response includes the prompt, so we need to decode properly
        full_response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        
        # Remove the prompt from the response
        response = full_response.replace(prompt, "").strip()
        
        # Only use fallback if truly empty
        if not response:
            response = "organize the scene"
        
        return response
    
    def _generate_subtask(self, image: Image.Image, objects: List[Dict], task: str) -> str:
        """Generate current subtask"""
        if not objects:
            return "explore the scene"
        
        # Use detected objects in prompt
        obj_list = ", ".join([obj["label"] for obj in objects[:3]])
        prompt = f"Task: {task}. Objects: {obj_list}. Current subtask:"
        
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_new_tokens=20, min_new_tokens=2)
        
        full_response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        response = full_response.replace(prompt, "").strip()
        
        # Only fallback if empty
        if not response:
            response = f"pick up {objects[0]['label']}"
        
        return response
    
    def _generate_action(self, image: Image.Image, objects: List[Dict], subtask: str) -> str:
        """Generate low-level action with position using AI"""
        if not objects:
            return "scan environment for objects"
        
        # Find relevant object for position
        target_obj = None
        for obj in objects:
            if obj["label"].lower() in subtask.lower():
                target_obj = obj
                break
        
        if not target_obj:
            target_obj = objects[0]
        
        # Use AI to generate action description
        x, y = target_obj["center"]
        prompt = f"Robot task: {subtask}. Target: {target_obj['label']} at ({x:.2f}, {y:.2f}). Describe gripper action:"
        
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_new_tokens=25, min_new_tokens=3)
        
        full_response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        action = full_response.replace(prompt, "").strip()
        
        # Fallback only if empty
        if not action:
            action = f"move gripper to {target_obj['label']} at position ({x:.2f}, {y:.2f})"
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
            ids = self.caption_model.generate(**inputs, max_new_tokens=10, min_new_tokens=1)
        
        full_response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        scene = full_response.replace(prompt, "").strip().lower()
        
        # Keep the AI response, only use fallback if empty
        if not scene:
            scene = "general"
        
        return scene
    
    def save_annotated_image(self, image_path: str, caption: VLACaption, output_path: str):
        """Save image with caption annotations overlay"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Color palette for different confidence levels
        def get_color(confidence):
            if confidence > 0.8:
                return (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                return (255, 255, 0)  # Yellow for medium confidence
            else:
                return (255, 0, 0)  # Red for low confidence
        
        # Try to use a better font if available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw bounding boxes and labels for detected objects
        for i, obj in enumerate(caption.objects[:10]):  # Show top 10 objects
            bbox = obj["bbox"]
            # Convert normalized coords to pixels
            x1 = int(bbox[0] * image.width)
            y1 = int(bbox[1] * image.height)
            x2 = int(bbox[2] * image.width)
            y2 = int(bbox[3] * image.height)
            
            color = get_color(obj["confidence"])
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw center point
            cx, cy = obj["center"]
            cx_px = int(cx * image.width)
            cy_px = int(cy * image.height)
            draw.ellipse([cx_px-5, cy_px-5, cx_px+5, cy_px+5], fill=color, outline=(255, 255, 255))
            
            # Draw label with background
            label = f"{obj['label']} ({obj['confidence']:.0%})"
            
            # Get text size for background
            bbox_text = draw.textbbox((x1, y1), label, font=font_small)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=(0, 0, 0, 200))
            
            # Draw text
            draw.text((x1+2, y1-text_height-2), label, fill=color, font=font_small)
            
            # Draw position coordinates
            pos_text = f"({cx:.2f}, {cy:.2f})"
            draw.text((cx_px+10, cy_px), pos_text, fill=(255, 255, 255), font=font_small)
        
        # Add caption information at the top with background
        info_lines = [
            f"Scene: {caption.scene_type}",
            f"Task: {caption.task_command}",
            f"Subtask: {caption.subtask}",
            f"Action: {caption.action_description[:50]}...",
            f"Confidence: {caption.confidence:.1%}",
            f"Objects Detected: {len(caption.objects)}"
        ]
        
        # Draw semi-transparent background for text
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, 0, image.width, 120], fill=(0, 0, 0, 180))
        
        # Composite overlay
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw info text
        y_offset = 10
        for line in info_lines:
            draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 18
        
        # Add spatial relations on the right side if present
        if caption.spatial_relations:
            y_offset = 130
            draw.text((10, y_offset), "Spatial Relations:", fill=(255, 255, 0), font=font)
            y_offset += 20
            for rel in caption.spatial_relations[:3]:
                draw.text((20, y_offset), f"• {rel}", fill=(200, 200, 200), font=font_small)
                y_offset += 15
        
        # Save annotated image
        image.save(output_path, 'PNG')
        print(f"✅ Annotated image saved as: {output_path}")
    
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
    import os
    
    # Initialize generator
    print("Initializing VLA Caption Generator...")
    generator = VLACaptionGenerator()
    
    # Process command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        task_context = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = sys.argv[3] if len(sys.argv) > 3 else "caption_output.json"
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
        output_file = "caption_output.json"
        print(f"Using example image: {image_path}")
    
    # Generate caption
    print(f"\nGenerating caption for: {image_path}")
    if task_context:
        print(f"Task context: {task_context}")
    
    caption = generator.generate_caption(image_path, task_context)
    
    # Save to JSON file
    output_data = {
        "image_path": image_path,
        "task_context": task_context,
        "caption": caption.to_dict(),
        "training_format": caption.to_training_format(),
        "hierarchical_format": caption.to_hierarchical_format()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Caption saved to: {output_file}")
    
    # Save annotated image
    annotated_file = output_file.replace('.json', '_annotated.png')
    if annotated_file == output_file:  # If no .json extension
        annotated_file = os.path.splitext(output_file)[0] + '_annotated.png'
    
    generator.save_annotated_image(image_path, caption, annotated_file)
    
    # Also save human-readable format
    txt_file = output_file.replace('.json', '.txt')
    with open(txt_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VLA CAPTION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Task Context: {task_context}\n\n")
        f.write(f"Task Command: {caption.task_command}\n")
        f.write(f"Subtask: {caption.subtask}\n")
        f.write(f"Action: {caption.action_description}\n")
        f.write(f"Scene: {caption.scene_type}\n")
        f.write(f"Confidence: {caption.confidence:.2%}\n\n")
        
        f.write(f"Objects Detected ({len(caption.objects)}):\n")
        for obj in caption.objects[:5]:
            f.write(f"  • {obj['label']}: pos({obj['center'][0]:.2f}, {obj['center'][1]:.2f}), ")
            f.write(f"size={obj['size']}, conf={obj['confidence']:.2%}\n")
        
        if caption.spatial_relations:
            f.write("\nSpatial Relations:\n")
            for rel in caption.spatial_relations:
                f.write(f"  • {rel}\n")
    
    print(f"✅ Human-readable output saved to: {txt_file}")
    
    # Display results
    print("\n" + "="*60)
    print("VLA CAPTION RESULTS")
    print("="*60)
    print(f"\nTask Command: {caption.task_command}")
    print(f"Subtask: {caption.subtask}")
    print(f"Action: {caption.action_description}")
    print(f"Scene: {caption.scene_type}")
    print(f"Confidence: {caption.confidence:.2%}")
    
    print(f"\nObjects Detected ({len(caption.objects)}):")
    for obj in caption.objects[:5]:
        print(f"  • {obj['label']}: pos({obj['center'][0]:.2f}, {obj['center'][1]:.2f}), "
              f"size={obj['size']}, conf={obj['confidence']:.2%}")
    
    if caption.spatial_relations:
        print("\nSpatial Relations:")
        for rel in caption.spatial_relations:
            print(f"  • {rel}")
    
    print(f"\n📸 Files created:")
    print(f"  - {output_file} (JSON data)")
    print(f"  - {annotated_file} (Annotated image)")
    print(f"  - {txt_file} (Human-readable text)")


if __name__ == "__main__":
    main()