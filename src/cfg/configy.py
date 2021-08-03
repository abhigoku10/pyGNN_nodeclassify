# -*- coding:utf-8 -*-
# author: abhigoku10

from pathlib import Path

from strictyaml import Bool, Float, Int, Map, Seq, Str, as_document, load


model_params = Map(
    {
        "model_architecture": Str(),
        "input_dim": Int(),
        "hidden": Int(),
        "dropout": Float(),
        "alpha": Float(),
        "nb_heads": Int(),

    }
)

dataset_params = Map(
    {
        "dataset_type": Str(),
        "save_results": Str(),
        
    }
)

train_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

val_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

test_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

train_params = Map(
    {
        "model_load_path": Str(),
        "model_save_path": Str(),
        "checkpoint_every_n_steps": Int(),
        "max_num_epochs": Int(),
        "patience": Int(),
        "lr_rate": Float(),
        "seed": Int(),
        "validationmode":Bool(),
        "weight_decay": Float(),
        "save_fig":Bool(),
        "save_log":Bool(),
        
     }
)

test_params = Map(
    {
        "model_load_path": Str(),
        "model_save_path": Str(),
        "output_viz": Bool()

     }
)


schema_v4 = Map(
    {
        "format_version": Int(),
        "model_params": model_params,
        "dataset_params": dataset_params,
        "train_data_loader": train_data_loader,
        "val_data_loader": val_data_loader,
        "test_data_loader": val_data_loader,
        "train_params": train_params,
        "test_params": test_params,
    }
)

SCHEMA_FORMAT_VERSION_TO_SCHEMA = {4: schema_v4}

def load_config_data(path: str) -> dict:
    yaml_string = Path(path).read_text()
    cfg_without_schema = load(yaml_string, schema=None)
    schema_version = int(cfg_without_schema["format_version"])
    if schema_version not in SCHEMA_FORMAT_VERSION_TO_SCHEMA:
        raise Exception(f"Unsupported schema format version: {schema_version}.")

    strict_cfg = load(yaml_string, schema=SCHEMA_FORMAT_VERSION_TO_SCHEMA[schema_version])
    cfg: dict = strict_cfg.data
    return cfg


def config_data_to_config(data):  # type: ignore
    return as_document(data, schema_v4)

def save_config_data(data: dict, path: str) -> None:
    cfg_document = config_data_to_config(data)
    with open(Path(path), "w") as f:
        f.write(cfg_document.as_yaml())


if __name__ == "__main__":
    # config_path=('E:\\Freelance_projects\\GNN\\Tuts\\pyGNN\\src\\config\\gcn_gen.yaml')

    config_path = ('E:\\Freelance_projects\\GNN\\Tuts\\pyGNN\\GCN\\config\\gcn_cora.yaml')
    
    configs = load_config_data(config_path)

