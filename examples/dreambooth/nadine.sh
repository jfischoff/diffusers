export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/jonathan/dreambooth/sks"
export CLASS_DIR="/home/jonathan/stable-diffusion-webui/models/dreambooth/nadine4/classifiers_0"
export OUTPUT_DIR="/home/jonathan/stable-diffusion-webui/models/Stable-diffusion/diffusers-nadine.ckpt"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a head shot of a sks woman" \
  --class_prompt="a head shot of a woman" \
  --resolution=512 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="polynomial" \
  --lr_warmup_steps=0 \
  --num_class_images=1000 \
  --max_train_steps=1300