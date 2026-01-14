import json
class returnRespVals:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "judgeString": ("STRING",),

      }
    }

  RETURN_TYPES = ("INT","INT","INT","INT",)
  RETURN_NAMES = ("totalCol", "totalRow", "bestRow", "bestCol")
  FUNCTION = "doit"
  CATEGORY = "Utils/resp"
  
  def doit(self, judgeString,):
    data = json.loads(judgeString)
    totalRow = int(data["gridTotal"]["x"])
    totalCol = int(data["gridTotal"]["y"])
    bestRow = int(data["bestImageGridPosition"]["x"])
    bestCol = int(data["bestImageGridPosition"]["y"])

    return (totalCol, totalRow, bestRow, bestCol)  
  
NODE_CLASS_MAPPINGS = {
    "returnRespVals": returnRespVals,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "returnRespVals": "returnRespVals",
}