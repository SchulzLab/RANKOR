#!/usr/bin/env python3
import argparse

from .io import read_gene_value_csv
from .predict import rank_drugs_from_profile


def main() -> None:
    p = argparse.ArgumentParser(description="RANKOR: rank drugs from a gene profile")
    p.add_argument("--input", required=True, help="CSV with columns: gene,value")
    p.add_argument("--train_h5ad", required=True, help="Reference AnnData (.h5ad) used for gene list and drug library")
    p.add_argument("--expression_encoder_ckpt", required=True, help="Transcriptome encoder checkpoint (.pth)")
    p.add_argument("--drug_encoder_ckpt", required=True, help="Drug ArcFace encoder checkpoint (.pth)")
    p.add_argument("--regressor_ckpt", required=True, help="Cross-space regressor checkpoint (.pth)")
    p.add_argument("--topk", type=int, default=0, help="Keep top-k drugs (0 = all)")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--print_top", type=int, default=20, help="Print first N ranked drugs to stdout (0 = none)")
    args = p.parse_args()

    profile = read_gene_value_csv(args.input)
    ranked = rank_drugs_from_profile(
        profile=profile,
        train_h5ad=args.train_h5ad,
        expression_encoder_ckpt=args.expression_encoder_ckpt,
        drug_encoder_ckpt=args.drug_encoder_ckpt,
        regressor_ckpt=args.regressor_ckpt,
        topk=args.topk,
    )

    ranked.to_csv(args.out_csv, index=False)
    if args.print_top and args.print_top > 0:
        print(ranked.head(args.print_top).to_string(index=False))


if __name__ == "__main__":
    main()

