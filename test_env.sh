python appearance_editing.py \
    --prompt "A cat sitting next to a mirror" \
    --edit_prompt "A tiger sitting next to a mirror" \
    --image_path "./data/example_images/gnochi_mirror.jpeg" \
    --output_name "gnochi_mirror_edited.png" \
    --amplify_word "tiger" \
    --amplify_weight 3 \
    --self_replace_step 0.7 0.8 0.9 \
    --blend_word1 "cat" \
    --blend_word2 "tiger" \
    --seed 1