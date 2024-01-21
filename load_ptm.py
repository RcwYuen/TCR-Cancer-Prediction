# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForMaskedLM
import argparse

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="PTM Downloader from HuggingFace"
    )
    parser.add_argument("-o", "--output-path", help="Path to Store PTMs")
    parser.add_argument("--silent", action="store_true", help="Report Progress")
    return parser.parse_args()

global SILENT, OUTPATH
if __name__ == "__main__":
    args = parse_command_line_arguments()
    SILENT = args.silent
    OUTPATH = "model" if args.output_path is None else args.output_path

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
    model = AutoModelForSequenceClassification.from_pretrained("wukevin/tcr-bert")
    model.save_pretrained(OUTPATH + "/ordinary/model/")
    tokenizer.save_pretrained(OUTPATH + "/ordinary/tokenizer/")
    tokenizer.save_vocabulary(OUTPATH + "/ordinary/tokenizer/")
    del tokenizer, model

    pipe = pipeline("text-classification", model="wukevin/tcr-bert")
    pipe.save_pretrained(OUTPATH + "/ordinary/pipe/")
    print ("Sample Input for Text Classification: 'Hello World!'")
    for outs in pipe("Hello World!", top_k = 10):
        print (f"{outs['label']:20}: {outs['score']:.4f}")
    print ("Check if output matches on")
    print ("https://huggingface.co/wukevin/tcr-bert?text=Hello+World%21\n")
    del pipe

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
    model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only")
    model.save_pretrained(OUTPATH + "/mlm-only/model/")
    tokenizer.save_pretrained(OUTPATH + "/mlm-only/tokenizer/")
    tokenizer.save_vocabulary(OUTPATH + "/mlm-only/tokenizer/")
    del tokenizer, model

    pipe = pipeline("fill-mask", model="wukevin/tcr-bert-mlm-only")
    pipe.save_pretrained(OUTPATH + "/mlm-only/pipe/")
    print ("Sample Input for Text Classification: 'Hello World.'")
    for outs in pipe("Hello World.", top_k = 10):
        print (f"{outs['token_str']:15}: {outs['score']:.4f}")
    print ("Check if output matches on")
    print ("https://huggingface.co/wukevin/tcr-bert-mlm-only?text=Hello+World.\n")
    del pipe

    print ("Done")

    