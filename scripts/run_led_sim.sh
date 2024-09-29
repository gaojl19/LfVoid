# LED-Simulated Environment
# Note: LED environments don't need dreambooth
export GPU_ID=0

# Note: You can change the input images and run editing
for i in {0..2}; 
do
python editing_appearance.py \
    --prompt "A red cylinder on a white table" \
    --edit_prompt "A green cylinder on a white table" \
    --image_path "./data/example_images/LED_sim/$i.png" \
    --experiment_name "LED_sim_$i" \
    --amplify_word "green" \
    --amplify_weight 3 \
    --self_replace_step 1.0 \
    --blend_word1 "red" \
    --blend_word2 "green" \
    --device "cuda:$GPU_ID" 
done

