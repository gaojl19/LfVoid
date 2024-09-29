# LED-Real Environment
# Note: LED environments don't need dreambooth
export GPU_ID=0

# Note: You can change the input images and run editing
for i in {0..2}; 
do
python editing_appearance.py \
    --prompt "A white table with a yellow duck light" \
    --edit_prompt "A white table with a dark duck light" \
    --image_path "./data/example_images/LED_real/$i.png" \
    --experiment_name "LED_real_$i" \
    --amplify_word "dark" \
    --amplify_weight 2 \
    --self_replace_step 0.1 \
    --blend_word1 "yellow" \
    --blend_word2 "dark" \
    --seed 0 \
    --device "cuda:$GPU_ID" 
done

