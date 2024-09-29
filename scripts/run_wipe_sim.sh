# Wipe-Simulated Environment
# Note: For each environment, you only need to run Dreambooth once

# Step 1. Dreambooth
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./data/dreambooth_inputs/Wipe_Sim"
export CLASS_DIR="./data/dreambooth_class/class_robot_arm"
export OUTPUT_DIR="./data/models/model_wipe_sim"
export GPU_ID=0

accelerate launch --gpu_ids $GPU_ID train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a sks robot arm" \
    --class_prompt="a robot arm" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=800 

# Step 2. null-text-ptp editing with dreambooth
# Note: you can change the input images and run Step 2 without re-running Step 1
for i in {0..2}; 
do
python editing_appearance.py \
    --prompt "A robot white table with markings on it" \
    --edit_prompt "A robot white table with nothing on it" \
    --image_path "./data/example_images/wipe_sim/$i.png" \
    --experiment_name "wipe_sim_$i" \
    --amplify_word "nothing" \
    --amplify_weight 3 \
    --self_replace_step 0 \
    --model_path $OUTPUT_DIR \
    --blend_word1 "markings" \
    --blend_word2 "nothing" \
    --device "cuda:$GPU_ID" 
done

