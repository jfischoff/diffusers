set -eux

# export CHECKPOINT_PATH="wavymulder/portraitplus"
workingDir="/home/jonathan/dreambooth/ryct-working-plus"
workingModelDir="wavymulder/portraitplus"
export INSTANCE_DIR="/home/jonathan/dreambooth/ryct"
export CLASS_DIR=/home/jonathan/dreambooth/ryct-working/class-images
export OUTPUT_DIR=$workingDir/output
loggingDir=$workingDir/logs
outputCheckpoint=/home/jonathan/stable-diffusion-webui/models/Stable-diffusion/portrait-plus-750.ckpt

seed=153214

perservationLoss=1.0

# Check if the workingModelDir exists
# if [ ! -d "$workingModelDir" ]; then
#   # Folder does not exist
#   echo "Folder $workingModelDir does not exist, extracting"
#   mkdir -p $workingModelDir
#   python3 ../../scripts/convert_original_stable_diffusion_to_diffusers.py \
#   --checkpoint_path $CHECKPOINT_PATH \
#   --prediction_type epsilon \
#   --dump_path $workingModelDir
# fi

# remove any mac hidden files
rm $INSTANCE_DIR/._* || true
rm $INSTANCE_DIR/.DS_Store || true

# TODO convert the checkpoints to ckpt format

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$workingModelDir  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR --with_prior_preservation \
  --prior_loss_weight=$perservationLoss \
  --instance_prompt="a photo of a ryct woman" \
  --class_prompt="a photo of a woman" \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="polynomial" \
  --lr_warmup_steps=0 \
  --num_class_images=1000 \
  --max_train_steps=750 \
  --seed=$seed \
  --checkpointing_steps=500 \
  --logging_dir=$loggingDir \
  --train_batch_size=2 

python3 ../../scripts/convert_diffusers_to_original_stable_diffusion.py \
  --checkpoint_path $outputCheckpoint \
  --model_path $OUTPUT_DIR