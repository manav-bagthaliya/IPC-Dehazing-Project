import os
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img,img2tensor
import argparse
import cv2
import glob
import os
import torch
from basicsr.utils import parallel_decode
from basicsr.archs.dehazeToken_arch import DehazeTokenNet,Critic
import numpy as np

# License plate detection imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

def detect_license_plates_opencv(image):
    """Detect license plates using OpenCV methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter by area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 2.0 < aspect_ratio < 5.5:  # License plate aspect ratio
                plates.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.8,
                    'method': 'opencv'
                })
    
    return plates

def detect_license_plates_yolo(image):
    """Detect license plates using YOLO if available"""
    if not YOLO_AVAILABLE:
        return []
    
    try:
        # Load pre-trained model (you can replace with custom license plate model)
        model = YOLO('yolov8n.pt')
        results = model(image, verbose=False)
        plates = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    plates.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf[0]),
                        'method': 'yolo'
                    })
        
        return plates
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []

def enhance_plate_region(image, plate_bbox):
    """Enhance the license plate region using proven character segmentation techniques"""
    x1, y1, x2, y2 = plate_bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    # Extract plate region
    plate_region = image[y1:y2, x1:x2]
    
    if plate_region.size == 0:
        return image
    
    # Apply proven license plate enhancement techniques
    
    # Step 1: Resize for consistent processing
    fixed_width = 400
    if plate_region.shape[1] != fixed_width:
        plate_region = cv2.resize(plate_region, (fixed_width, int(plate_region.shape[0] * fixed_width / plate_region.shape[1])))
    
    # Step 2: Convert to HSV and extract V channel for better text processing
    hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    
    # Step 3: Apply adaptive thresholding (proven method)
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)  # Invert for better character visibility
    
    # Step 4: Apply morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Apply gentle Gaussian blur to reduce noise
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    # Step 6: Apply unsharp masking for text sharpness
    gaussian = cv2.GaussianBlur(thresh, (0, 0), 1.0)
    thresh = cv2.addWeighted(thresh, 1.3, gaussian, -0.3, 0)
    
    # Step 7: Convert back to BGR for display
    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Step 8: Apply gentle contrast enhancement
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
    
    # Step 9: Apply bilateral filter to smooth while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)
    
    # Ensure values are in valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def process_with_license_plate_detection(img_tensor, output_img_tensor, device):
    """Process the dehazed image with license plate enhancement"""
    # Convert tensors to numpy arrays
    output_img_np = tensor2img(output_img_tensor, rgb2bgr=True)
    original_img_np = tensor2img(img_tensor, rgb2bgr=True)
    
    # Detect license plates using multiple methods
    plates = []
    
    # Try YOLO first (more accurate)
    if YOLO_AVAILABLE:
        plates.extend(detect_license_plates_yolo(original_img_np))
    
    # Fallback to OpenCV method
    if not plates:
        plates.extend(detect_license_plates_opencv(original_img_np))
    
    # Remove duplicates and sort by confidence
    unique_plates = []
    seen_bboxes = set()
    
    for plate in sorted(plates, key=lambda x: x['confidence'], reverse=True):
        bbox = tuple(plate['bbox'])
        # Check if this bbox overlaps significantly with existing ones
        is_duplicate = False
        for seen_bbox in seen_bboxes:
            if bbox_overlap(bbox, seen_bbox) > 0.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_plates.append(plate)
            seen_bboxes.add(bbox)
    
    # Enhance each detected license plate
    enhanced_count = 0
    for plate in unique_plates:
        bbox = plate['bbox']
        enhanced_plate = enhance_plate_region(output_img_np, bbox)
        
        # Replace the enhanced plate back
        x1, y1, x2, y2 = bbox
        if 0 <= y1 < y2 <= output_img_np.shape[0] and 0 <= x1 < x2 <= output_img_np.shape[1]:
            output_img_np[y1:y2, x1:x2] = enhanced_plate
            enhanced_count += 1
    
    if enhanced_count > 0:
        print(f"Enhanced {enhanced_count} license plate(s)")
    
    # Convert back to tensor
    return img2tensor(output_img_np)

def bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    return intersection / min(area1, area2)

def enhance_output_quality(image):
    """Apply balanced quality enhancement to the dehazed output"""
    try:
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply gentle CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply very subtle contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)
        
        # Apply gentle bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing output quality: {e}")
        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_path',type=str,default='pretrained_models/predictor.pth')
    parser.add_argument('--critic_path',type=str,default=  'pretrained_models/critic.pth')
    parser.add_argument('-i','--input', type=str, default='examples', help='input test image folder')
    parser.add_argument('-o','--output', type=str, default='results', help='output test image folder')
    parser.add_argument('-n', type=int, default=8, help='num_iterations')
    parser.add_argument('--max_size', type=int, default=1500, help='max_size')
    parser.add_argument('--resize_to', type=int, default=0, help='If > 0, upscale inputs whose long edge is smaller than this value to this size before processing')
    parser.add_argument('--force_width', type=int, default=0, help='If > 0 and used with --force_height, resize ALL inputs to this width before processing')
    parser.add_argument('--force_height', type=int, default=0, help='If > 0 and used with --force_width, resize ALL inputs to this height before processing')
    parser.add_argument('--enhance_plates', action='store_true', help='Enable license plate enhancement')
    
    args = parser.parse_args()
    
    num_iterations = args.n
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16, LQ_stage=True, predictor_name='swinLayer').to(device)
    net_g.load_state_dict(torch.load(args.predictor_path)['params'], strict=True)
    net_g.eval()
    
    net_critic= Critic().to(device)
    net_critic.load_state_dict(torch.load(args.critic_path)['params'], strict=True)
    net_critic.eval()

    def tokens_to_logits(seq: torch.Tensor,h=0,w=0,critic=False) -> torch.Tensor:
        if critic:
            logits=net_critic(seq,h,w)
        else:
            logits = net_g.transformer(seq,critic)
        return logits 
    
    def tokens_to_feats(seq:torch.Tensor)->torch.Tensor:
        '''
        seq :b, 1, h, w
        '''
        feats = net_g.vqgan.quantize.get_codebook_entry(seq.long())
        return feats

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
   
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        pbar.set_description(f'Test {idx} {imgname}')
        img = cv2.imread(path, cv2.IMREAD_COLOR)/255.0
        img=img2tensor(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            _,_,origin_h,origin_w = img.shape

            # Determine resizing strategy
            forced = (args.force_width > 0 and args.force_height > 0)
            if forced:
                # Resize ALL inputs to the forced target size
                img = torch.nn.UpsamplingBilinear2d((args.force_height, args.force_width))(img)
            else:
                # Optionally upscale small images to a target long-edge size
                if args.resize_to and max(origin_h, origin_w) < args.resize_to:
                    scale = args.resize_to / max(origin_h, origin_w)
                    img = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img)

                # Downscale overlarge images for compute/memory
                if origin_h*origin_w >= args.max_size**2:
                    scale = args.max_size/max(origin_h, origin_w)
                    img = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img)
    
            # padding to multiple of window_size * 8
            wsz = 32
            _, _, h_old, w_old = img.shape
            h_pad = (h_old // wsz + 1) * wsz - h_old
            w_pad = (w_old // wsz + 1) * wsz - w_old
            img = torch.cat([img, torch.flip(img, [2])],
                            2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])],
                            3)[:, :, :, :w_old + w_pad]
    
            ################### Encoder #####################
            enc_feats = net_g.vqgan.multiscale_encoder(img.detach())
            enc_feats = enc_feats[::-1]

            x = enc_feats[0]
            b, c, h, w = x.shape

            feat_to_quant = net_g.vqgan.before_quant(x)
            _,_,lq_tokens=net_g.vqgan.encode(img)
            
            mask_tokens=-1*torch.ones(b,h*w).to(feat_to_quant.device).long()

            output_tokens,mask_tokens = parallel_decode.decode_critic(
                mask_tokens,
                feat_to_quant,
                tokens_to_logits,
                tokens_to_feats,
                num_iter=num_iterations,
                )
            
            quant_feats=tokens_to_feats(output_tokens[:, -1,:].reshape(b,1,h,w))

                # ################## Generator ####################
            after_quant_feat = net_g.vqgan.after_quant(quant_feats)

            x=after_quant_feat
            for i in range(net_g.max_depth):
                cur_res = net_g.gt_res // 2**net_g.max_depth * 2**i
                x = net_g.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
                x = net_g.vqgan.decoder_group[i](x)
                
            output_img = net_g.vqgan.out_conv(x)
            output_img = output_img[..., :h_old , :w_old ]
            
            # License plate enhancement
            if args.enhance_plates:
                output_img = process_with_license_plate_detection(img, output_img, device)
            
            # Preserve consistent output size and quality
            if forced:
                # Ensure exact forced output size
                output_img = torch.nn.UpsamplingBilinear2d((args.force_height, args.force_width))(output_img)
            else:
                # If we downscaled due to exceeding max_size, upscale back to original size
                if origin_h*origin_w >= args.max_size**2:
                    output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                # If we upscaled small images, keep the upscaled size for better quality
                elif args.resize_to and max(origin_h, origin_w) < args.resize_to:
                    # Keep the upscaled size for better quality
                    pass
                else:
                    # Ensure we return to original size if no resizing was done
                    if output_img.shape[2:] != (origin_h, origin_w):
                        output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                
            # Save with high quality settings
            output_img_np = tensor2img(output_img, rgb2bgr=True)
            
            # Apply final quality enhancement to the output
            output_img_np = enhance_output_quality(output_img_np)
            
            # Save with high quality settings
            output_path = os.path.join(args.output, f'{imgname}.png')
            cv2.imwrite(output_path, output_img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Save resized input for reference so user can see the processing size
            resized_inputs_dir = os.path.join(args.output, 'resized_inputs')
            os.makedirs(resized_inputs_dir, exist_ok=True)
            cv2.imwrite(os.path.join(resized_inputs_dir, f'{imgname}.png'), tensor2img(img, rgb2bgr=True))


        pbar.update(1)
           
    pbar.close()


if __name__ == '__main__':
    main()
