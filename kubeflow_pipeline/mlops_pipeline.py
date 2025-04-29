import kfp
from kfp import dsl
import kfp.compiler
import kfp.compiler.compiler


## Pipeline Components

def data_processing_ops():
    return dsl.ContainerOp(
        name= "Data Processing",
        image="taufeeqai/my-mlops-app:latest",
        command = ["python","src/data_processing.py"]
    )

def model_training_ops():
    return dsl.ContainerOp(
        name= "Model Training",
        image="taufeeqai/my-mlops-app:latest",
        command = ["python","src/model_trainer.py"]
    )


## ML pipeline 

@dsl.pipeline(
    name="ML Pipeline",
    description="Colorectal cancer patient survival prediction kubeflow pipeline"  
)

def ml_pipeline():
    data_processing = data_processing_ops()
    model_training = model_training_ops().after(data_processing)


## Run the pipeline

if __name__=="__main__":
    kfp.compiler.Compiler().compile(
        ml_pipeline, "ml_pipeline.yaml"
    )