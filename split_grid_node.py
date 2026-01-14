# import torch 

class gridreturnsingle:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "row": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
        "column": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
        "x_input": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
        "y_input": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("images",)
  FUNCTION = "doit"
  CATEGORY = "Utils/image"

  def doit(self, images, row, column, x_input, y_input):
    """
    images: tensor shape (batch, height, width, channels)
    row, column: grid size (rows x columns)
    x_input, y_input: 1-based column (x) and row (y) indices of the cell to return
    """
    # defensive clamp of indices (convert to 0-based)
    x_idx = max(1, min(column, int(x_input))) - 1
    y_idx = row - max(1, min(row, int(y_input)))

    batch, height, width, channels = images.shape

    # cell size (use integer division). Last row/column will include any remainder pixels.
    cell_w = width // column
    cell_h = height // row

    start_x = x_idx * cell_w
    start_y = y_idx * cell_h

    # ensure last column/row take remaining pixels so we don't drop pixels due to rounding
    end_x = width if (x_idx == column - 1) else (start_x + cell_w)
    end_y = height if (y_idx == row - 1) else (start_y + cell_h)

    # slice out the requested cell
    cropped = images[:, start_y:end_y, start_x:end_x, :]

    return (cropped,)


NODE_CLASS_MAPPINGS = {
    "gridreturnsingle": gridreturnsingle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "gridreturnsingle": "gridreturnsingle",
}