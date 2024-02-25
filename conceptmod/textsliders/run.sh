#!/bin/bash

# Sample entries to process along with user provided entries
SAMPLE_ENTRIES=(
    #"|happy|sad"
    #"|laughing|crying"
    #"|attractive|ugly"
    #"|action shot|long exposure",
    
    #"|rpg portrait|photograph",
    #"|angry|calm",
    #"|disagree|agree",
    #"|sleeping|awake",
    #"|cute|rough",
    #"|action shot|long exposure",

    #"|monster|person"
    #"|eyes buldging|eyes closed"
    #"|coffee shop|bar"

    #"|christmas|"
    #"|christmas|monday"
    #"|santa|devil"

    "|the grinch|whos from whoville",
    "|final boss|level 1 mob",
    "|good|evil",
    "|angel|demon",
    "|superhero|supervillain",

    #"movie still|movie still, action shot| movie still, long exposure",
    #"photo|photo, 3d render| photo, cartoon"
    #"movie still|movie still, eye contact| movie still, looking away",
    #"movie still|movie still, cute| movie still, scary",
    #"movie still|movie still, adorable| movie still, scary",
    #"person|person waving hello|person looking away",
    #"woman|woman, laughing|woman, stern"
)

# Process sample entries
for entry in "${SAMPLE_ENTRIES[@]}"; do
    python make_config.py "$entry"
    CUDA_VISIBLE_DEVICES=1 python train_lora_xl.py --attributes 'woman, man, bright, dim, sunny, dark, cartoon, photo' --name "$entry" --rank 4 --alpha 1 --config_file data/config-xl.yaml
done
