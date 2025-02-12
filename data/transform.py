from PIL import Image

class CropTransform:
    """
    Crop 1/3 & Transform.
    use_crop -> True ==> Cropped img, False ==> Full img
    resize_size: (width, height).
    """
    def __init__(self, resize_size, use_crop=True):
        self.resize_size = resize_size
        self.use_crop = use_crop

    def __call__(self, img: Image.Image):
        if self.use_crop:
            width, height = img.size
            ## bottom 1/3 -> crop
            img = img.crop((0, height // 3, width, height))
        ## Resize
        img = img.resize(self.resize_size)
        return img
