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
import os
import sys
import urllib.request
import tempfile
from urllib.parse import urlparse

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
        self.caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            device_map="auto" if DEVICE == 'cuda' else None
        )
        if DEVICE == 'cpu':
            self.caption_model = self.caption_model.to(DEVICE)
        
        print("Loading object detection model...")
        self.detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detector_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        ).to(DEVICE)
        
        print("Models loaded successfully!")
    
    def generate_caption(self, image_path: str, task_context: Optional[str] = None) -> VLACaption:
        """Generate complete VLA caption for an image"""
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
            
            center = [(norm_bbox[0] + norm_bbox[2]) / 2, (norm_bbox[1] + norm_bbox[3]) / 2]
            
            objects.append({
                "label": self.detector_model.config.id2label[label.item()],
                "bbox": norm_bbox,
                "center": center,
                "confidence": float(score.item()),
                "size": self._get_size(norm_bbox)
            })
        
        return sorted(objects, key=lambda x: x['confidence'], reverse=True)[:10]
    
    def _get_size(self, bbox: List[float]) -> str:
        """Estimate object size from bbox"""
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return "large" if area > 0.2 else "medium" if area > 0.05 else "small"
    
    def _generate_with_prompt(self, image: Image.Image, prompt: str, max_tokens: int, min_tokens: int = 1) -> str:
        """Helper method to generate text with BLIP-2"""
        inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = self.caption_model.generate(**inputs, max_new_tokens=max_tokens, min_new_tokens=min_tokens)
        
        full_response = self.caption_processor.batch_decode(ids, skip_special_tokens=True)[0]
        return full_response.replace(prompt, "").strip()
    
    def _generate_task_command(self, image: Image.Image, context: Optional[str]) -> str:
        """Generate high-level task with context-aware prompts"""
        # Context-specific prompts for better identification
        if context:
            context_lower = context.lower()
            if "ingredient" in context_lower or "food" in context_lower:
                prompt = "What specific ingredients or food items are shown? Be precise:"
            elif "identify" in context_lower or "recognize" in context_lower:
                prompt = "Identify the specific objects in this image:"
            elif "vegetable" in context_lower or "produce" in context_lower:
                prompt = "What vegetables or produce items are visible? Name them specifically:"
            elif "cook" in context_lower or "prep" in context_lower:
                prompt = "What cooking ingredients need to be prepared? Be specific:"
            else:
                prompt = f"This is a {context} task. What should be done? Give one short command:"
        else:
            prompt = "What task should be performed here? Give one short command:"
        
        response = self._generate_with_prompt(image, prompt, 20, 2)
        return response or "analyze the scene"
    
    def _generate_subtask(self, image: Image.Image, objects: List[Dict], task: str) -> str:
        """Generate current subtask with better object awareness"""
        if not objects:
            # Ask BLIP-2 directly what's in the image
            prompt = "What is the main object or item in this image? Be specific:"
            item = self._generate_with_prompt(image, prompt, 15, 1)
            return f"identify {item}" if item else "explore the scene"
        
        obj_list = ", ".join([obj["label"] for obj in objects[:3]])
        
        # If task involves identification, be more specific
        if "identify" in task.lower() or "ingredient" in task.lower():
            prompt = f"Looking at this image more carefully, what specific type of {obj_list} is this?"
        else:
            prompt = f"Task: {task}. Objects detected: {obj_list}. What specific action is needed?"
        
        response = self._generate_with_prompt(image, prompt, 20, 2)
        return response or f"examine {objects[0]['label']}"
    
    def _generate_action(self, image: Image.Image, objects: List[Dict], subtask: str) -> str:
        """Generate low-level action with position using AI"""
        if not objects:
            # Try to get BLIP-2 to describe what it sees
            prompt = "Describe the position of the main object in the image:"
            description = self._generate_with_prompt(image, prompt, 25, 3)
            return description or "scan environment for objects"
        
        # Find relevant object
        target_obj = next((obj for obj in objects if obj["label"].lower() in subtask.lower()), objects[0])
        
        x, y = target_obj["center"]
        
        # More descriptive action generation
        prompt = f"For {subtask}, describe how to interact with the {target_obj['label']} at position ({x:.2f}, {y:.2f}):"
        
        action = self._generate_with_prompt(image, prompt, 30, 3)
        
        if not action:
            action = f"move gripper to {target_obj['label']} at position ({x:.2f}, {y:.2f})"
            grip_type = " with precision grip" if target_obj["size"] == "small" else \
                       " with wide grip" if target_obj["size"] == "large" else ""
            action += grip_type
        
        return action
    
    def _extract_spatial_relations(self, objects: List[Dict]) -> List[str]:
        """Extract spatial relationships"""
        relations = []
        
        for i, obj1 in enumerate(objects[:4]):
            for obj2 in objects[i+1:5]:
                rel = self._compute_relation(obj1, obj2)
                if rel:
                    relations.append(rel)
                if len(relations) >= 5:
                    return relations
        
        return relations
    
    def _compute_relation(self, obj1: Dict, obj2: Dict) -> Optional[str]:
        """Compute spatial relation between two objects"""
        c1, c2 = obj1["center"], obj2["center"]
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        
        # Check vertical relations
        if abs(dx) < 0.1:
            if dy > 0.1:
                return f"{obj1['label']} above {obj2['label']}"
            elif dy < -0.1:
                return f"{obj1['label']} below {obj2['label']}"
        
        # Check horizontal relations
        if abs(dy) < 0.1:
            if dx > 0.1:
                return f"{obj1['label']} left of {obj2['label']}"
            elif dx < -0.1:
                return f"{obj1['label']} right of {obj2['label']}"
        
        # Check proximity
        if np.sqrt(dx**2 + dy**2) < 0.15:
            return f"{obj1['label']} near {obj2['label']}"
        
        return None
    
    def _identify_scene(self, image: Image.Image) -> str:
        """Identify scene type with more descriptive output"""
        prompt = "Describe the main items visible in this image in 3-4 words:"
        scene = self._generate_with_prompt(image, prompt, 15, 1).lower()
        return scene or "general scene"
    
    def save_annotated_image(self, image_path: str, caption: VLACaption, output_path: str):
        """Save image with caption annotations overlay"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Color mapping based on confidence
        get_color = lambda conf: (0, 255, 0) if conf > 0.8 else (255, 255, 0) if conf > 0.6 else (255, 0, 0)
        
        # Load fonts
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = font_small = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for obj in caption.objects[:10]:
            bbox = obj["bbox"]
            x1, y1 = int(bbox[0] * image.width), int(bbox[1] * image.height)
            x2, y2 = int(bbox[2] * image.width), int(bbox[3] * image.height)
            cx_px, cy_px = int(obj["center"][0] * image.width), int(obj["center"][1] * image.height)
            
            color = get_color(obj["confidence"])
            
            # Draw rectangle and center point
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.ellipse([cx_px-5, cy_px-5, cx_px+5, cy_px+5], fill=color, outline=(255, 255, 255))
            
            # Draw label with background
            label = f"{obj['label']} ({obj['confidence']:.0%})"
            bbox_text = draw.textbbox((x1, y1), label, font=font_small)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=(0, 0, 0))
            draw.text((x1+2, y1-text_height-2), label, fill=color, font=font_small)
            
            # Draw position
            draw.text((cx_px+10, cy_px), f"({obj['center'][0]:.2f}, {obj['center'][1]:.2f})", 
                     fill=(255, 255, 255), font=font_small)
        
        # Create overlay for info panels
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, 0, 450, 120], fill=(0, 0, 0, 200))
        
        # Top right spatial relations panel
        if caption.spatial_relations:
            overlay_draw.rectangle([image.width-250, 0, image.width, 100], fill=(0, 0, 0, 200))
        
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw info text (top left) - adjusted for longer scene descriptions
        info_lines = [
            f"Scene: {caption.scene_type[:40]}",
            f"Task: {caption.task_command[:40]}",
            f"Subtask: {caption.subtask[:40]}",
            f"Action: {caption.action_description[:40]}...",
            f"Confidence: {caption.confidence:.1%}",
            f"Objects: {len(caption.objects)}"
        ]
        
        for i, line in enumerate(info_lines):
            draw.text((10, 10 + i * 18), line, fill=(255, 255, 255), font=font)
        
        # Draw spatial relations (top right)
        if caption.spatial_relations:
            draw.text((image.width-240, 10), "Spatial Relations:", fill=(255, 255, 0), font=font)
            for i, rel in enumerate(caption.spatial_relations[:4]):
                draw.text((image.width-240, 30 + i * 15), f"> {rel}", fill=(200, 200, 200), font=font_small)
        
        image.save(output_path, 'PNG')
        print(f"Annotated image saved: {output_path}")
    
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
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} captions to {output_file}")
        return results


def download_image(url: str) -> str:
    """Download image from URL and return local path, supports multiple formats"""
    # Extension mapping
    ext_map = {
        '.jpg': ['.jpg', '.jpeg'],
        '.png': ['.png'],
        '.webp': ['.webp'],
        '.gif': ['.gif'],
        '.bmp': ['.bmp']
    }
    
    # Determine extension from URL
    url_lower = url.lower()
    ext = '.jpg'  # default
    
    for extension, patterns in ext_map.items():
        if any(pattern in url_lower for pattern in patterns):
            ext = extension
            break
    
    # If no extension found in URL, try content-type
    if ext == '.jpg' and not any(p in url_lower for patterns in ext_map.values() for p in patterns):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                
                type_map = {
                    'jpeg': '.jpg', 'jpg': '.jpg',
                    'png': '.png', 'webp': '.webp',
                    'gif': '.gif', 'bmp': '.bmp'
                }
                
                for key, val in type_map.items():
                    if key in content_type:
                        ext = val
                        break
                
                # Download the content
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(response.read())
                    return tmp.name
        except:
            pass
    
    # Standard download
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(response.read())
            return tmp.name


def main():
    """Example usage and test"""
    # Initialize generator
    print("Initializing VLA Caption Generator...")
    generator = VLACaptionGenerator()
    
    # Process command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python VlaCaptionGenerator.py <image_path_or_url> [task_context] [output_file]")
        print("\nExamples:")
        print('  python VlaCaptionGenerator.py image.jpg "ingredient identification"')
        print('  python VlaCaptionGenerator.py https://example.com/garlic.jpg "identify vegetable"')
        print("\nTask Context Suggestions:")
        print("  - 'ingredient identification' - for identifying specific food items")
        print("  - 'identify vegetable' - for vegetable recognition")
        print("  - 'food preparation' - for cooking prep tasks")
        print("  - 'recognize produce' - for produce identification")
        print("  - 'identify object' - for general object identification")
        sys.exit(1)
    
    image_path = sys.argv[1]
    task_context = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else "caption_output.json"
    
    # Check if input is URL
    if image_path.startswith(('http://', 'https://')):
        print(f"Downloading image from: {image_path}")
        image_path = download_image(image_path)
        print(f"Downloaded to: {image_path}")
    
    # Generate caption
    print(f"\nGenerating caption for: {image_path}")
    if task_context:
        print(f"Task context: {task_context}")
    
    caption = generator.generate_caption(image_path, task_context)
    
    # Save outputs
    output_base = os.path.splitext(output_file)[0]
    
    # JSON output with UTF-8 encoding
    output_data = {
        "image_path": image_path,
        "task_context": task_context,
        "caption": caption.to_dict(),
        "training_format": caption.to_training_format(),
        "hierarchical_format": caption.to_hierarchical_format()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nCaption saved: {output_file}")
    
    # Annotated image
    annotated_file = f"{output_base}_annotated.png"
    generator.save_annotated_image(image_path, caption, annotated_file)
    
    # Text output with UTF-8 encoding
    txt_file = f"{output_base}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
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
            f.write(f"  > {obj['label']}: pos({obj['center'][0]:.2f}, {obj['center'][1]:.2f}), ")
            f.write(f"size={obj['size']}, conf={obj['confidence']:.2%}\n")
        
        if caption.spatial_relations:
            f.write("\nSpatial Relations:\n")
            for rel in caption.spatial_relations:
                f.write(f"  > {rel}\n")
    
    print(f"Text output saved: {txt_file}")
    
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
        print(f"  > {obj['label']}: pos({obj['center'][0]:.2f}, {obj['center'][1]:.2f}), "
              f"size={obj['size']}, conf={obj['confidence']:.2%}")
    
    if caption.spatial_relations:
        print("\nSpatial Relations:")
        for rel in caption.spatial_relations:
            print(f"  > {rel}")
    
    print(f"\nFiles created:")
    print(f"  - {output_file} (JSON data)")
    print(f"  - {annotated_file} (Annotated image)")
    print(f"  - {txt_file} (Human-readable text)")


if __name__ == "__main__":
    main()