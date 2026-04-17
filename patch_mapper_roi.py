import re

with open('src/mapper.py', 'r') as f:
    content = f.read()

old_code = """
                rendered_exposed = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
                static_mask = (~cleanup_mask).float()
                static_mask_3 = static_mask.unsqueeze(0).expand_as(rendered_exposed)
                static_denom = static_mask_3.sum().clamp(min=1.0)
"""

new_code = """
                rendered_exposed = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
                static_mask = (~cleanup_mask).float()
                
                roi_radius = int(temporal_params.get("post_cleanup_roi_radius", 0))
                if roi_radius > 0:
                    roi_mask = temporal_utils.max_pool_spatial_map(cleanup_mask.float(), roi_radius) > 0.5
                    static_mask = static_mask * roi_mask.float()
                    
                static_mask_3 = static_mask.unsqueeze(0).expand_as(rendered_exposed)
                static_denom = static_mask_3.sum().clamp(min=1.0)
"""

content = content.replace(old_code, new_code)
with open('src/mapper.py', 'w') as f:
    f.write(content)
