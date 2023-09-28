from dataclasses import dataclass

import json

from argparse import ArgumentParser

from datasets import load_dataset


@dataclass
class EvaluateExisting:
    hf_path: str
    subset: str
    split: str

    def run(self) -> None:
        d = load_dataset(
            path=self.hf_path,
            name=self.subset,
            split=self.split,
        )
        for example in d:
            prompt = example["prompt"]
            reference_function_name, reference_args_dict = json.loads(
                example["reference"]
            )
            accuracy = example["correct"]

            prediction = json.loads(example["prediction"])
            if isinstance(prediction, str):
                prediction_str = f"Error message\n\t{prediction}\n\n"
            else:
                predicted_function_name, predicted_args_dict = prediction
                prediction_str = f"Prediction\n\t{predicted_function_name}, {predicted_args_dict}\n\n"

            accuracy_str = "CORRECT" if accuracy else "WRONG"
            print(
                f"Prompt\n\t{prompt}\n\nReference\n\t{reference_function_name}, {reference_args_dict}\n\n{prediction_str}{accuracy_str}\n{'-' * 80}\n"
            )

        accuracy = 100 * sum(d["correct"]) / len(d)
        print(f"Accuracy: {accuracy:.3f}%")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    EvaluateExisting(**vars(args)).run()
