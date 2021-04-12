import numpy as np
import time

from functions.utilities import Utilities

import torch

class Evaluation():
    def __init__(self):
        self.utils = Utilities()
        self.device = self.utils.select_processing_unit()
    
    def train_model(self, model, train_loader, valid_loader, criterion, optimizer, 
                    batch_size, train_data_size, file_path, epochs, iterations = 2, 
                    patience = 10):
        start_time = time.time()
        train_loss, valid_loss = 0, 0
        early_stop_count = 0
        early_stop = False
        min_valid_loss = np.Inf
        step_total = int(np.ceil(train_data_size / batch_size))
        # Point when training switches to validation
        step_count = step_total // iterations
        training_losses, validation_losses = [], []

        for epoch in range(epochs):
            steps = 0
            model.train() # activating training

            if early_stop:
                break

            for images, labels in train_loader:
                steps += 1
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Vanishing gradient problem
                optimizer.zero_grad()
                output, aux_output = model.forward(images)

                # Comparing predicted values against true values
                loss1 = criterion(output, labels)
                loss2 = criterion(aux_output, labels)
                loss = loss1 + (0.4 * loss2)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if steps % step_count == 0:
                    model.eval() # activates validation
                    # Turns off all gradients to prevents weight from being updated (to speed up inferences / reduce memory consumption)
                    with torch.no_grad():
                        accuracy, valid_loss = self.validate_model(model, valid_loader, criterion)

                    train_loss /= step_count
                    valid_loss /= len(valid_loader)
                    accuracy /= len(valid_loader)

                    training_losses.append(train_loss)
                    validation_losses.append(valid_loss)

                    print(f"\tEpoch: {epoch + 1}/{epochs} | Step: {steps}/{step_total}",
                          f" | Training Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}",
                          f" | Accuracy: {accuracy * 100:.2f}%")

                    early_stop = self.utils.early_stop_check(valid_loss, patience, 
                                                             early_stop, early_stop_count)

                    if valid_loss <= min_valid_loss:
                        print(f"\tThe validation loss has changed from {min_valid_loss:.3f}",
                              f" to {valid_loss:.3f}.", end=" ")
                        min_valid_loss = valid_loss
                        self.utils.save_model(model, file_path, training_losses, 
                                              validation_losses)
                        early_stop_count = 0
                    elif early_stop:
                        break
                    else:
                        early_stop_count += 1
                        print(f"\tEarly stop counter: {early_stop_count}/{patience}")  

                    train_loss = 0
                    model.train()

        self.utils.time_taken(time.time() - start_time)

    def validate_model(self, model, valid_loader, criterion):
        accuracy, valid_loss = 0, 0
        for images, labels in valid_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = model.forward(images)
            # Comparing predicted values against true values
            loss = criterion(output, labels)
            valid_loss += loss.item()

            class_probs = torch.exp(output)
            highest_class_prob = (labels.data == class_probs.max(dim = 1)[1])

            accuracy += highest_class_prob.type_as(torch.FloatTensor()).mean()

        return accuracy, valid_loss

    def test_model(self, model, model_name, test_loader, top_k_pred_count = 3):
        # top_k_pred_count = The number of k predictions (default 3) per image (performance metric)
        with torch.no_grad():
            model.to(self.device)
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output, _ = model.forward(images)
                probabilities = torch.exp(output)
                # k_preds = Classes of strongest k predictions for each image
                _, k_preds = probabilities.topk(top_k_pred_count, dim = 1)
                # top_preds = The strongest prediction for each image
                top_preds = probabilities.max(dim = 1)[1]

                performance_metrics = self._calc_performance_metrics(model_name, top_preds, 
                                                                     labels, k_preds)
                return performance_metrics
    
    def _calc_performance_metrics(self, model_name, top_preds, labels, k_preds):
        precision, recall = self._calc_precision_recall(top_preds, labels)
        correct_pred_count = top_preds.eq(labels).sum().item()
        accuracy = correct_pred_count / len(labels)
        top1_error = 1 - accuracy

        top_preds = top_preds.to(self.device)
        k_preds = k_preds.to(self.device)
        k_correct_count = 0

        for prediction in range(len(labels)):
            pred_bools = k_preds[prediction].eq(labels[prediction])
            if True in pred_bools:
                k_correct_count += 1

        k_error = 1 - (k_correct_count / len(labels))
        f1_score = (2 * precision * recall) / (precision + recall)

        performance_metrics = {
            "Name": model_name,
            "Accuracy": accuracy,
            "Top 1 Error": top1_error,
            f"Top {len(k_preds[0])} Error": k_error,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }

        return performance_metrics

    def _calc_precision_recall(self, top_preds, labels):
        # Calculate confusion matrix metrics
        tp, tn, fp, fn = 0, 0, 0, 0
        for label in range(len(labels)):
            if (labels[label] == top_preds[label]) == 1:
                tp += 1   
            if top_preds[label] == 1 and labels[label] != top_preds[label]:
                fp += 1

            if (labels[label] == top_preds[label]) == 0:
                tn += 1
            if top_preds[label] == 0 and labels[label] != top_preds[label]:
                fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return precision, recall