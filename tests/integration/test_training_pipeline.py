import pytest
import pytorch_lightning as pl

from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule
from text2cypher.finetuning.models.t5_model import T5NoteGenerationModel

@pytest.mark.integration
def test_training_pipeline_runs_and_validates():
    datamodule = NoteChatDataModule(
        model_name="t5-small",
        batch_size=2,
        max_length=128,
        num_workers=0,
        train_samples=4,
        val_samples=2,
    )

    model = T5NoteGenerationModel(
        model_name="t5-small",
        model_type="t5",
        learning_rate=1e-4,
        use_quantization=False,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(model, datamodule=datamodule)

    assert trainer.global_step > 0  # confirms training loop ran

    # Optional: run validation manually and check metrics
    val_metrics = trainer.validate(model=model, datamodule=datamodule)
    assert "val_loss" in val_metrics[0]
