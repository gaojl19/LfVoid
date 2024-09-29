# Push-Simulated Environment
# Note: For each environment, you only need to run Dreambooth once

# Step 1. Dreambooth
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./data/dreambooth_inputs/Push_Sim"
export CLASS_DIR="./data/dreambooth_class/class_redcube"
export OUTPUT_DIR="./data/models/model_push_sim"
export GPU_ID=0

accelerate launch --gpu_ids $GPU_ID train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of a sks cube" \
  --class_prompt="a photo of a red cube" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800


# Step 2. null-text-ptp-dd editing with dreambooth
# Note: you can change the input images and run Step 2 without re-running Step 1
for i in {0..2}; 
do
python editing_structure.py \
    --prompt "A photo of a red sks cube and a gripper on a white table" \
    --image_path "./data/example_images/push_sim/$i.png" \
    --experiment_name "push_sim_$i" \
    --region 0.5 0.7 0.7 0.9 \
    --noise_level 0.5 \
    --prompt_indice 6 7 \
    --num_trailing_maps 20 \
    --editing_steps 10 \
    --annealing_coef 0.01 \
    --annealing_threshold 0 \
    --use_base_edit true \
    --plot_region \
    --model_path $OUTPUT_DIR \
    --device "cuda:$GPU_ID" 
done