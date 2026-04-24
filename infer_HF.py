import sys
import argparse
from huggingface_hub import snapshot_download
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True, help="HuggingFace Hub Repo ID (e.g. Vladimirescu/WCA-Satellite)")
    parser.add_argument("--lat", type=float, default=-52.055073, help="Latitude")
    parser.add_argument("--lon", type=float, default=-59.717834, help="Longitude")
    parser.add_argument("--osm_path", type=str, default=None, help="Path to the OSM map. If lat/lon are given, this argument is ignored.")
    parser.add_argument("--prompt", type=str, default="A depressing city.", help="Text prompt")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    print(f"Downloading codebase and model weights from {args.hf_repo}...")
    # downloads Hugging Face repo
    repo_dir = snapshot_download(repo_id=args.hf_repo)
    # Point Python to use the dynamically downloaded Hugging Face files
    sys.path.insert(0, repo_dir)
    from pipeline import SatSynthPipeline
    
    print("Loading pipeline...")
    # Load the pipeline from HF
    pipe = SatSynthPipeline.from_pretrained(
        args.hf_repo, 
        fallback_local=True, 
        model_path=f"{repo_dir}/model.safetensors",
        config_path=f"{repo_dir}/config.yaml"
    )
    
    print("Running Inference")
    if args.osm_path is not None:
        osm_map = Image.open(args.osm_path)
    else:
        osm_map = None

    result = pipe(
        prompt=args.prompt, 
        lat=args.lat, 
        lon=args.lon,
        osm_map=osm_map,
        return_osm=True
    )
    
    result.images[0].save("generated.png")
    result.images[1].save("osm_map.png")
