from get_obj_det_models import get_florence2_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_detection_model, obj_detection_processor = get_florence2_model(device)
