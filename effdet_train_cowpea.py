import argparse
import os
import torch
import wandb
import cv2
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import ImageFile
from loader import build_loader, EfficientDetDataModule, annotate_image
from model import EfficientDet

hyperparameter_defaults = dict(
    learning_rate = 0.003373,
    iou_threshold = 0.3056,
    step_size = 2.261,
    gamma = 4.27,
    weight_decay = 0.08205,
    conf_threshold = 0.1072
)

def main(args, config):

    # Initialize logger and save directory
    logger = WandbLogger()
    save_dir = os.path.join(args.save,args.name)

    # Make the data loader and module
    loader = build_loader(args.root)
    num_classes = loader.num_classes
    dm = EfficientDetDataModule(loader=loader,
                                num_workers=args.workers,
                                batch_size=args.batch,
                                img_size=args.img_size
    )

    # Training callbacks
    model_checkpoint = ModelCheckpoint(save_top_k = 2,
                                       monitor='val_map',
                                       mode='max',
                                       verbose=True,
                                       filename=os.path.join(save_dir, '{epoch}_{val_map:.4f}'))
    
    # Make the model
    model = EfficientDet(config,
                        num_classes=num_classes,
                        image_size=args.img_size,
                        orig_size=[512, 512],
                        architecture='efficientdet_d0'
    )

    # Train the model
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    trainer = pl.Trainer(logger=logger,
                         max_epochs=args.epochs,
                         callbacks=[model_checkpoint],
                         gpus=args.gpus if torch.cuda.is_available() else 0,
                         log_every_n_steps=2
    )
    trainer.fit(model=model, datamodule=dm)

    # Run the test loop and visualize predictions
    trainer.test(datamodule = dm)

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, f'model.pth'))

    # Run a sample prediction
    model.switch_predict()
    indices = np.random.randint(0,100,10)
    image_path = [os.path.join(args.root,'images',image) for image in os.listdir(os.path.join(args.root, 'images'))]
    images = [image_path[i] for i in indices]
    predictions = model.predict(images, args.img_size)

    # Create wandb table and visualize
    predictions_artifact = wandb.Artifact(args.name, type="predictions")
    columns = ['image', 'predicted']
    data = []
    for i in range(len(images)):
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted = annotate_image(images[i], predictions[i][0])
        data.append([wandb.Image(image),wandb.Image(predicted)])
    table = wandb.Table(columns=columns, data=data)
    predictions_artifact.add(table, 'predictions')
    wandb.run.log_artifact(predictions_artifact)
    wandb.log({'predictions': table})

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-r','--root', type=str, required=True,
                    help='Path to directory containing images.')
    ap.add_argument('-e','--epochs', type=int, default=100,
                    help='Max epochs for training.')
    ap.add_argument('-i','--img-size', type=int, default=512,
                    help='Max epochs for training.')
    ap.add_argument('-w','--workers', type=int, default=5,
                    help='Number of dataloader workers per GPU.')
    ap.add_argument('-b','--batch', type=int, default=4,
                    help='Batch size per GPU.')
    ap.add_argument('-g','--gpus', type=int, default=1,
                    help='Number of GPUs')
    ap.add_argument('-l','--logger', type=str, required=True,
                    help='Project name in WandB.')
    ap.add_argument('-n','--name', type=str, required=True,
                    help='Name of run in WandB.')
    ap.add_argument('-s','--save', type=str, required=True,
                    help='Directory to save weights')
    args, unknown = ap.parse_known_args()

    # initialize wandb run
    wandb.init(project=args.logger, entity='paibl', name=args.name, config=hyperparameter_defaults)
    config = wandb.config

    main(args, config)