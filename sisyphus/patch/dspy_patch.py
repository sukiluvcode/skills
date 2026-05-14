"""Import this to overwrite dspy default save method, since original save method does not support pydantic model"""
import dspy
import ujson
from pydantic import BaseModel

def custom_save(self, path, save_field_meta=False):
    def convert_to_dict(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(i) for i in obj]
        elif isinstance(obj, dspy.Example):
            d = dict(obj)
            return convert_to_dict(d)
        else:
            return obj

    state = self.dump_state(save_field_meta)
    state = convert_to_dict(state)
    with open(path, "w") as f:
        f.write(ujson.dumps(state, indent=2, ensure_ascii=False))

dspy.Module.save = custom_save
