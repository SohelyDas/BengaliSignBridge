import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "isl_lstm.keras"


def project_status():
	samples = list(DATA_DIR.glob("**/*.npy")) if DATA_DIR.is_dir() else []
	print(f"Project: {ROOT_DIR}")
	print(f"Training samples: {len(samples)}")
	print(f"Trained model: {'yes' if MODEL_PATH.exists() else 'no'}")
	if not samples:
		print("Next step: python collect_data.py")
	elif not MODEL_PATH.exists():
		print("Next step: python train.py")
	else:
		print("Training is complete. Use the trained model from your application code.")


def run_script(script_name):
	script_path = ROOT_DIR / script_name
	result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT_DIR)
	raise SystemExit(result.returncode)


def main():
	parser = argparse.ArgumentParser(description="Bengali Sign Bridge project launcher")
	parser.add_argument(
		"command",
		nargs="?",
		choices=("check", "collect", "train"),
		default="check",
		help="check the setup, collect camera samples, or train the model",
	)
	args = parser.parse_args()

	if args.command == "collect":
		run_script("collect_data.py")
	elif args.command == "train":
		run_script("train.py")
	else:
		project_status()


if __name__ == "__main__":
	main()
