import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

from PIL import Image
from PIL.PngImagePlugin import PngImageFile

class Blur:
    """Face blurring as a meant for privacy.
    There are two types of blurring: pixelation and gaussian blur.
    """
    def __init__(self, facedetection_path, blocks: str = 10, wider: int = 0, blurmode: str = "face", resolution: str = "normal", kernel_size: int = 23, confidence: float = 0.7):
        super().__init__()

        from ultralytics import YOLO

        self.model = YOLO(facedetection_path)
        self.blurmode = blurmode
        self.resolution = resolution
        self.blocks = blocks
        self.offset = wider if wider else 0
        self.confidence = confidence
        self.gaussian_kernel = (kernel_size, kernel_size) if kernel_size else (23, 23)
    
    
    def pixelate_region(self, im, box, pixel_size=10):
        """Pixelate the region defined by the bounding box."""
        # Extract the region to be pixelated
        box = tuple(map(int, box))
        region = im.crop(box)

        new_width = max(1, int(region.size[0] // pixel_size))
        new_height = max(1, int(region.size[1] // pixel_size))
        
        # Resize the region to a smaller size
        small = region.resize((new_width, new_height), resample=Image.NEAREST)
        
        # Resize back to the original size
        pixelated_region = small.resize(region.size, Image.NEAREST)
        
        # Paste the pixelated region back into the image
        im.paste(pixelated_region, box)

    def draw_rectangles_and_pixelate(self, im, bounding_boxes, pixel_num):
        # Open an image file
        
        draw = ImageDraw.Draw(im)
        
        # Iterate through bounding boxes and pixelate each region
        for box in bounding_boxes:
            # Pixelate the region inside the bounding box
            if box[-2] > 0.8:
                pixel_size = ((box[2] - box[0] + box[3] - box[1])/2)/pixel_num
                self.pixelate_region(im, box[:-2], pixel_size=pixel_size)
        
        return im
    
    def corrupt(self, image: PngImageFile) -> PngImageFile:
        bboxes = self.model.predict(image, verbose=False)
        w, h = image.size

        try:
            bbox = None
            self.block = None
            
            if self.blurmode == "face":
                bbox = bboxes[0].boxes.data.cpu().numpy()
                w =  bbox[0, 2] - bbox[0, 0]
                h = bbox[0, 3] - bbox[0, 1]
                self.blocks = int(w/50) if self.resolution == "low" else int(w/20)
                self.blocks = max(1, self.blocks)

            elif self.blurmode == "full":
                bbox = np.array([[0.0, 0.0, w, h, 1.0, 0]])
                self.blocks = int(w/50) if self.resolution == "low" else int(w/20)
                self.blocks = max(1, self.blocks)

            else:
                raise ValueError(f"Mode {self.mode} not recognized.")
        
            corr_image = self.draw_rectangles_and_pixelate(image, bbox, pixel_num=self.blocks)
            return corr_image
        except:
            return image
    
    def forward(self, *args, **kwargs):
        return self.corrupt(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



