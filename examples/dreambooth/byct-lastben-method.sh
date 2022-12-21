export MODEL_NAME="/home/jonathan/stable-diffusion-webui/models/dreambooth/jonathan2"
export INSTANCE_DIR="/home/jonathan/dreambooth/byct"
export OUTPUT_DIR="/home/jonathan/stable-diffusion-webui/models/Stable-diffusion/relighting"

Seed=9
PT=""
Text_Training_Steps=350
Unet_Training_Steps=6000
stpsv=1000
save_n_steps=500
precision=no
Res=512
SESSION_DIR=

accelerate launch train_dreambooth.py \
  --image_captions_filename \
  --train_text_encoder \
  --dump_only_text_encoder \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$PT" \
  --seed=$Seed \
  --resolution=512 \
  --mixed_precision=$precision \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="polynomial" \
  --lr_warmup_steps=0 \
  --max_train_steps=$Text_Training_Steps

accelerate launch train_dreambooth.py \
  $Style \
  --image_captions_filename \
  --train_only_unet \
  --save_starting_step=$stpsv \
  --save_n_steps=$stp \
  --Session_dir=$SESSION_DIR \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$PT" \
  --seed=$Seed \
  --resolution=$Res \
  --mixed_precision=$precision \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="polynomial" \
  --lr_warmup_steps=0 \
  --max_train_steps=$Unet_Training_Steps