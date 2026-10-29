"""GroundLens command-line interface."""
import argparse
import json
from .data import load_benchmark
from .retrieval import Retriever
from .answering import answer
from .evaluation import evaluate, choose_threshold


def main(argv=None):
    parser=argparse.ArgumentParser(prog='groundlens')
    sub=parser.add_subparsers(dest='command', required=True)
    ev=sub.add_parser('evaluate', help='Run dev-calibrated held-out benchmark')
    ev.add_argument('--out-dir', default='reports')
    ev.add_argument('--max-dev-fpr', type=float, default=.1)
    ask=sub.add_parser('ask', help='Return an extractive quote and source ID or abstain')
    ask.add_argument('question')
    ask.add_argument('--method', choices=['bm25','tfidf','hybrid'], default='tfidf')
    args=parser.parse_args(argv)
    if args.command=='evaluate':
        result=evaluate(args.out_dir, max_dev_fpr=args.max_dev_fpr)
        print(json.dumps(result, indent=2))
    else:
        docs, queries=load_benchmark()
        retriever=Retriever(docs)
        dev=[q for q in queries if q.split=='dev']
        ranking={q.id:retriever.rank(q.text, args.method) for q in dev}
        cal=choose_threshold(dev, ranking)
        print(json.dumps(answer(args.question, retriever.rank(args.question,args.method),cal['threshold']), indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
