import glob
import os
import pandas as pd

from .experiment import run_experiment


def main():
    positive_datasets = glob.glob(os.path.join("kinase_low", "*.csv"))

    if not positive_datasets:
        print("No positive datasets found in kinase_low directory")
        return

    all_results = {}
    augmentation_report = []

    for rare_kinase_csv in positive_datasets:
        dataset_name = os.path.basename(rare_kinase_csv)
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 80}")

        results = run_experiment(rare_kinase_csv)
        all_results[dataset_name] = results

        if results:
            for multiplier, multiplier_results in results.items():
                if 'augmentation_stats' in multiplier_results:
                    stats = multiplier_results['augmentation_stats']
                    stats['dataset'] = dataset_name
                    augmentation_report.append(stats)

        print("\n" + "=" * 100 + "\n")

    if augmentation_report:
        report_df = pd.DataFrame(augmentation_report)
        os.makedirs("result_low6", exist_ok=True)
        report_path = os.path.join("result_low6", "augmentation_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Augmentation report saved to {report_path}")

    print("\n\nSummary of Results:")
    print("=" * 80)
    for dataset, multiplier_results in all_results.items():
        print(f"\nDataset: {dataset}")
        for multiplier, results in multiplier_results.items():
            print(f"  Multiplier: {multiplier * 100}%")
            print(f"    Baseline AUC: {results['baseline']['test_results']['auc']:.4f}")
            print(f"    Initial AUC: {results['initial']['auc']:.4f}")
            print(f"    Improvement: {results['initial']['auc'] - results['baseline']['test_results']['auc']:.4f}")
            print(f"    Baseline ACC: {results['baseline']['test_results']['acc']:.4f}")
            print(f"    Initial ACC: {results['initial']['test_results']['acc']:.4f}")
            print(
                f"    ACC Improvement: "
                f"{results['initial']['test_results']['acc'] - results['baseline']['test_results']['acc']:.4f}"
            )
            print(
                f"    Raw Generated Samples Average Quality: "
                f"{results['augmentation_stats'].get('raw_avg_quality', 0):.4f}"
            )
            print(
                f"    Selected Samples Average Quality: "
                f"{results['augmentation_stats'].get('selected_avg_quality', 0):.4f}"
            )

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()