import torch
import tqdm

def get_feature_extractor(model_str):
    from transformers import CLIPVisionModel, AutoModel, AutoProcessor
    if model_str.startswith('openai/clip-'):
        model = CLIPVisionModel.from_pretrained(model_str, output_hidden_states=True)
        processor = AutoProcessor.from_pretrained(model_str)
        processor = processor.image_processor
    else:
        model = AutoModel.from_pretrained(model_str, output_hidden_states=True)
        processor = AutoProcessor.from_pretrained(model_str)
    return model, processor

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

@torch.inference_mode()
def extract_features(loader, model, index, device='cuda'):
    all_features, all_labels = [], []
    for data, labels, paths in tqdm.tqdm(loader):
        w, h = data.shape[-2:]
        data = data.reshape(-1, 3, w, h).to(device)
        _, intermediates = model.forward_intermediates(data, return_prefix_tokens=True, indices=[index])
        _, cls_tokens = zip(*intermediates)
        all_features.append(cls_tokens[0].squeeze(1).cpu())
        all_labels.append(labels.reshape(-1))
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)