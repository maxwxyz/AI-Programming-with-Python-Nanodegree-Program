from utilities import get_predict_args

model, class_to_idx = load_checkpoint(args.checkpoint_path)

device = torch.device('cuda:0' if torch.cuda.is_available() and args.device else 'cpu'

probs, classes = predict(args.image_path, model, args.topk, class_to_idx)

print ('Classes: ', classes)
print('Probability: ', probs)