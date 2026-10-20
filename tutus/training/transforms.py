from torchvision import transforms



class ResizeShortSideMax:
    def __init__(self, max_short):
        self.max_short = max_short

    def __call__(self, img):
        w, h = img.size
        short = min(w, h)

        if short <= self.max_short:
            return img  # No resizing needed

        # Compute new size preserving aspect ratio
        scale = self.max_short / short
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        return transforms.Resize((new_h, new_w))(img)
    

class ResizeLongSideMax:
    def __init__(self, max_long):
        self.max_long = max_long

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)

        if long <= self.max_long:
            return img  # No resizing needed

        # Scale so the longest side becomes max_long
        scale = self.max_long / long
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        return transforms.Resize((new_h, new_w))(img)
