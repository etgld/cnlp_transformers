import logging
from collections import defaultdict
from itertools import chain, groupby
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import tqdm
from transformers import EvalPrediction

from .cnlp_processors import classification, relex, tagging

logger = logging.getLogger(__name__)

Cell = Tuple[int, int, int]


def restructure_prediction(
    task_names: List[str],
    raw_prediction: EvalPrediction,
    max_seq_length: int,
    tagger: Dict[str, bool],
    relations: Dict[str, bool],
    output_prob: bool,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Dict[str, Tuple[int, int]],
]:
    task_label_ind = 0

    # disagreement collection stuff for this scope
    task_label_to_boundaries: Dict[str, Tuple[int, int]] = {}
    task_label_to_label_packet: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    for task_ind, task_name in enumerate(task_names):
        preds, labels, pad, prob_values = structure_labels(
            raw_prediction,
            task_name,
            task_ind,
            task_label_ind,
            max_seq_length,
            tagger,
            relations,
            task_label_to_boundaries,
            output_prob,
        )
        task_label_ind += pad

        task_label_to_label_packet[task_name] = (preds, labels, prob_values)
    return (
        task_label_to_label_packet,
        task_label_to_boundaries,
    )


def structure_labels(
    p: EvalPrediction,
    task_name: str,
    task_ind: int,
    task_label_ind: int,
    max_seq_length: int,
    tagger: Dict[str, bool],
    relations: Dict[str, bool],
    task_label_to_boundaries: Dict[str, Tuple[int, int]],
    output_prob: bool,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # disagreement collection stuff for this scope

    pad = 0
    prob_values = np.ndarray([])
    labels = np.ndarray([])
    if tagger[task_name]:
        preds = np.argmax(p.predictions[task_ind], axis=2)
        # labels will be -100 where we don't need to tag
    elif relations[task_name]:
        preds = np.argmax(p.predictions[task_ind], axis=3)
    else:
        preds = np.argmax(p.predictions[task_ind], axis=1)
        if output_prob:
            prob_values = np.max(p.predictions[task_ind], axis=1)

    # for inference
    if not hasattr(p, "label_ids") or p.label_ids is None:
        return preds, np.array([]), pad, np.array([])
    if relations[task_name]:
        # relation labels
        labels = p.label_ids[
            :, :, task_label_ind : task_label_ind + max_seq_length
        ].squeeze()
        task_label_to_boundaries[task_name] = (
            task_label_ind,
            task_label_ind + max_seq_length,
        )
        pad = max_seq_length
    elif p.label_ids.ndim == 3:
        if tagger[task_name]:
            labels = p.label_ids[:, :, task_label_ind : task_label_ind + 1].squeeze()
        else:
            labels = p.label_ids[:, 0, task_label_ind].squeeze()
        task_label_to_boundaries[task_name] = (task_label_ind, task_label_ind + 1)
        pad = 1
    elif p.label_ids.ndim == 2:
        labels = p.label_ids[:, task_ind].squeeze()

    return preds, labels, pad, prob_values


def remove_newline(review):
    review = review.replace("&#039;", "'")
    review = review.replace("\n", " <cr> ")
    review = review.replace("\r", " <cr> ")
    review = review.replace("\t", " ")
    return review


def compute_disagreements(
    preds: np.ndarray,
    labels: np.ndarray,
    output_mode: str,
) -> np.ndarray:
    """
    Function that defines and computes the metrics used for each task.
    When adding a task definition to this file, add a branch to this
    function defining what its evaluation metric invocation should be.
    If the new task is a simple classification task, a sensible default
    is defined; falling back on this will trigger a warning.

    :param str task_name: the task name used to index into cnlp_processors
    :param numpy.ndarray preds: the predicted labels from the model
    :param numpy.ndarray labels: the true labels
    :rtype: typing.Dict[str, typing.Any]
    :return: a dictionary containing evaluation metrics
    """

    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if output_mode == classification:
        return classification_disagreements(preds=preds, labels=labels)
    elif output_mode == tagging:
        return tagging_disagreements(preds=preds, labels=labels)
    elif output_mode == relex:
        return relation_disagreements(
            preds=preds,
            labels=labels,
        )
    else:
        raise Exception("As yet unsupported task in cnlpt")


def classification_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where(np.not_equal(preds, labels))
    return indices


def tagging_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where([any(neqs) for neqs in np.not_equal(preds, labels)])
    return indices


def relation_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where([neqs.any() for neqs in np.not_equal(preds, labels)])
    return indices


def process_prediction(
    task_names: List[str],
    error_analysis: bool,
    output_prob: bool,
    character_level: bool,
    task_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    eval_dataset,
    task_to_label_space: Dict[str, List[str]],
    output_mode: Dict[str, str],
) -> pd.DataFrame:
    task_to_error_inds: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
    if error_analysis:
        for task, label_packet in tqdm.tqdm(
            task_to_label_packet.items(), desc="computing disagreements"
        ):
            preds, labels, prob_values = label_packet
            task_to_error_inds[task] = compute_disagreements(
                preds, labels, output_mode[task]
            )

        relevant_indices: Iterable[int] = set(
            map(int, chain.from_iterable(task_to_error_inds.values()))
        )

    else:
        relevant_indices = range(len(eval_dataset["text"]))

    classification_tasks = filter(
        lambda t: output_mode[t] == classification, task_names
    )

    tagging_tasks = filter(lambda t: output_mode[t] == tagging, task_names)

    relex_tasks = filter(lambda t: output_mode[t] == relex, task_names)

    # ordering in terms of ease of reading
    out_table = pd.DataFrame(
        columns=["text", *classification_tasks, *tagging_tasks, *relex_tasks],
        index=relevant_indices,
    )

    out_table["text"] = [eval_dataset["text"][index] for index in relevant_indices]
    out_table["text"] = out_table["text"].apply(remove_newline)

    out_table["text"] = out_table["text"].str.replace('"', "")
    out_table["text"] = out_table["text"].str.replace("//", "")
    out_table["text"] = out_table["text"].str.replace("\\", "")
    word_ids = eval_dataset["word_ids"]
    for task_name, packet in tqdm.tqdm(
        task_to_label_packet.items(), desc="getting human readable labels"
    ):
        preds, labels, prob_values = packet
        if not output_prob:
            prob_values = np.array([])
        task_labels = task_to_label_space[task_name]
        error_inds = task_to_error_inds[task_name]
        target_inds = error_inds if len(error_inds) > 0 else relevant_indices
        out_table[task_name][target_inds] = get_output_list(
            error_analysis=error_analysis,
            character_level=character_level,
            prob_values=prob_values,
            pred_task=task_name,
            task_labels=task_labels,
            prediction=preds,
            labels=labels,
            output_mode=output_mode,
            error_inds=error_inds,
            word_ids=word_ids,
            text_column=out_table["text"],
        )
    return out_table


# might be more efficient to return a pd.Series or something for the
# assignment and populate it via a generator but for now just use a list
def get_output_list(
    error_analysis: bool,
    character_level: bool,
    prob_values: np.ndarray,
    pred_task: str,
    task_labels: List[str],
    prediction: np.ndarray,
    labels: Union[None, np.ndarray],
    output_mode: Dict[str, str],
    error_inds: np.ndarray,
    word_ids: List[List[int]],
    text_column: pd.Series,
) -> List[str]:
    if len(error_inds) > 0 and error_analysis:
        relevant_prob_values = (
            prob_values[error_inds]
            if output_mode[pred_task] == classification and len(prob_values) > 0
            else prob_values
        )
        ground_truth = labels[error_inds].astype(int)
        task_prediction = prediction[error_inds].astype(int)
        text_samples = pd.Series(text_column[error_inds])
    else:
        relevant_prob_values = prob_values
        ground_truth = labels.astype(int) if error_analysis else None
        task_prediction = prediction.astype(int)
        text_samples = text_column
    task_type = output_mode[pred_task]
    if task_type == classification:
        return get_classification_prints(
            pred_task, task_labels, ground_truth, task_prediction, relevant_prob_values
        )

    elif task_type == tagging:
        return get_tagging_prints(
            character_level,
            pred_task,
            task_labels,
            ground_truth,
            task_prediction,
            text_samples,
            word_ids,
        )
    elif task_type == relex:
        return get_relex_prints(
            pred_task, task_labels, ground_truth, task_prediction, word_ids
        )
    else:
        return len(error_inds) * ["UNSUPPORTED TASK TYPE"]


def get_classification_prints(
    task_name: str,
    classification_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    prob_values: np.ndarray,
) -> List[str]:
    predicted_labels = [classification_labels[index] for index in task_predictions]

    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        if ground == predicted:
            return f"_no_{task_name}_error_"
        return f"Ground: {ground} , Predicted: {predicted}"

    pred_list = predicted_labels
    if ground_truths is not None:
        ground_strings = [classification_labels[index] for index in ground_truths]

        pred_list = [*map(clean_string, zip(ground_strings, predicted_labels))]

    if len(prob_values) == len(predicted_labels):
        return [
            f"{pred} , Probability {prob:.6f}"
            for pred, prob in zip(pred_list, prob_values)
        ]
    return pred_list


def get_tagging_prints(
    character_level: bool,
    task_name: str,
    tagging_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    text_samples: pd.Series,
    word_ids: List[List[Union[None, int]]],
) -> List[str]:
    resolved_predictions = task_predictions

    # to save ourselves the branch instructions
    # in all the nested functions
    get_tokens = lambda: None
    token_sep = ""  # default since typesystem doesn't like the None
    if character_level:
        get_tokens = lambda inst: [token for token in inst if token is not None]
        token_sep = ""
    else:
        get_tokens = lambda inst: [char for char in inst.split() if char is not None]
        token_sep = " "

    def flatten_dict(d):
        def tups(k, ls):
            return ((k, elem) for elem in ls)

        return chain.from_iterable(
            (((k, span) for k, span in tups(key, spans)) for key, spans in d.items())
        )

    def dict_to_str(d, tokens):
        result = " , ".join(
            f'{key}: "{token_sep.join(tokens[span[0]:span[1]])}"'
            for key, span in flatten_dict(d)
        )
        return result

    def group_and_span(inds: List[int]) -> List[Tuple[int, int]]:
        ranges = []
        for _, group in groupby(enumerate(inds), lambda x: x[0] - x[1]):
            group = [g[1] for g in group]
            # adjusted for python list conventions
            ranges.append((group[0], group[-1] + 1))
        return ranges

    # since sometimes it's just
    # BIO with no suffixes and
    # we'll need to use the column name
    def get_ner_type(tag: str) -> str:
        elems = tag.split("-")
        if len(elems) > 1:
            return elems[-1].lower()
        return task_name.lower()

    def types2spans(raw_tag_inds: np.ndarray, word_ids: List[Union[None, int]]):
        type2inds = defaultdict(list)
        relevant_token_ids_and_tags = [
            (word_id, tag)
            for word_id, tag in zip(word_ids, raw_tag_inds)
            if word_id is not None
        ]
        relevant_token_ids_and_tags = [
            next(group)
            for _, group in groupby(relevant_token_ids_and_tags, key=lambda s: s[0])
        ]

        raw_labels = [tagging_labels[tag] for _, tag in relevant_token_ids_and_tags]
        for index, raw_label in enumerate(raw_labels):
            if raw_label != "O":
                type2inds[get_ner_type(raw_label)].append(index)

        return {ner_type: group_and_span(inds) for ner_type, inds in type2inds.items()}

    def dictmerge(
        ground_dict: Dict[str, List[Tuple[int, int]]],
        pred_dict: Dict[str, List[Tuple[int, int]]],
    ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        disagreements: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for key in {*ground_dict.keys(), *pred_dict.keys()}:
            ground_spans = ground_dict[key] if key in ground_dict.keys() else []

            pred_spans = pred_dict[key] if key in pred_dict.keys() else []

            ground_not_in_pred = [
                span for span in ground_spans if span not in pred_spans
            ]

            pred_not_in_ground = [
                span for span in pred_spans if span not in ground_spans
            ]

            disagreements["ground"][key].extend(ground_not_in_pred)

            disagreements["predicted"][key].extend(pred_not_in_ground)

        return disagreements

    def get_error_out_string(
        disagreements: Dict[str, Dict[str, List[Tuple[int, int]]]], instance: str
    ) -> str:
        instance_tokens = get_tokens(instance)
        ground_string = (
            dict_to_str(disagreements["ground"], instance_tokens)
            if "ground" in disagreements.keys()
            else ""
        )

        predicted_string = (
            dict_to_str(disagreements["predicted"], instance_tokens)
            if "predicted" in disagreements.keys()
            else ""
        )

        if len(ground_string) == 0 == len(predicted_string):
            return f"_no_{task_name.lower()}_errors_"

        return f"Ground: {ground_string} Predicted: {predicted_string}"

    def get_pred_out_string(
        type2spans: Dict[str, List[Tuple[int, int]]], instance: str
    ):
        instance_tokens = get_tokens(instance)
        return dict_to_str(type2spans, instance_tokens)

    pred_span_dictionaries = (
        types2spans(pred, word_id_ls)
        for pred, word_id_ls in zip(resolved_predictions, word_ids)
    )
    if ground_truths is not None:
        ground_span_dictionaries = (
            types2spans(ground_truth, word_id_ls)
            for ground_truth, word_id_ls in zip(ground_truths, word_ids)
        )
        disagreement_dicts = (
            dictmerge(ground_dictionary, pred_dictionary)
            for ground_dictionary, pred_dictionary in zip(
                ground_span_dictionaries, pred_span_dictionaries
            )
        )

        # returning list instead of generator since pandas needs that
        return [
            get_error_out_string(disagreements, instance)
            for disagreements, instance in zip(disagreement_dicts, text_samples)
        ]

    return [
        get_pred_out_string(type_2_pred_spans, instance)
        for type_2_pred_spans, instance in zip(pred_span_dictionaries, text_samples)
    ]


def get_relex_prints(
    task_name: str,
    relex_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    word_ids: List[List[Union[None, int]]],
) -> List[str]:
    resolved_predictions = task_predictions
    none_index = relex_labels.index("None") if "None" in relex_labels else -1

    def relevant_elements(
        mat_row: np.ndarray, word_id_ls: List[Union[None, int]], word_id: int
    ) -> List[int]:
        relevant_token_ids_and_tags = [
            (word_id, cell_value)
            for word_id, cell_value in zip(word_id_ls, mat_row)
            if word_id is not None
        ]

        relevant_token_ids_and_tags = (
            list(group)[0]
            for _, group in groupby(relevant_token_ids_and_tags, key=lambda s: s[0])
        )
        relevant_token_ids, relevant_tags = zip(*relevant_token_ids_and_tags)
        if word_id in {*relevant_token_ids}:
            return list(relevant_tags)
        return []

    # thought we'd filtered them out but apparently not
    def tuples_to_str(label_tuples: Iterable[Cell]) -> str:
        print(label_tuples)
        return " ".join(
            f"( {row}, {col}, {relex_labels[label]} )"
            for row, col, label in sorted(label_tuples)
        )

    def normalize_cells(
        raw_cells: np.ndarray, token_ids: List[Union[None, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        (invalid_inds,) = np.where(np.diag(raw_cells) != -100)
        # just in case
        reduced_matrix = np.array(
            [
                relevant_elements(mat_row, token_ids, word_id)
                for word_id, mat_row in enumerate(raw_cells)
                if len(relevant_elements(mat_row, token_ids, word_id)) > 0
            ]
        )

        np.fill_diagonal(reduced_matrix, none_index)

        assert (
            reduced_matrix.shape[0] == reduced_matrix.shape[1]
        ), f"reduced matrix shape: {reduced_matrix.shape}"
        return invalid_inds, reduced_matrix

    def find_disagreements(
        ground_pair: Tuple[np.ndarray, np.ndarray],
        pred_pair: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[Iterable[Cell], Iterable[Cell], Iterable[Cell]]:
        invalid_ground_inds, ground_matrix = ground_pair

        _, pred_matrix = pred_pair

        disagreements = np.where(ground_matrix != pred_matrix)

        if len(ground_matrix) == 0 == len(pred_matrix) == len(invalid_ground_inds):
            return [], [], []

        bad_cells = (
            (
                (*i, j)
                for i, j in zip(
                    zip(invalid_ground_inds, invalid_ground_inds),
                    ground_matrix[invalid_ground_inds, invalid_ground_inds],
                )
            )
            if len(invalid_ground_inds) > 0
            else []
        )
        # nones will just clutter things up
        # and we will be able to infer disagreements on nones
        # from each other
        ground_cells = filter(
            lambda t: t[-1] != none_index,
            zip(*disagreements, ground_matrix[disagreements]),
        )

        pred_cells = filter(
            lambda t: t[-1] != none_index,
            zip(*disagreements, pred_matrix[disagreements]),
        )

        return bad_cells, ground_cells, pred_cells

    def to_error_string(
        bad_cells: Iterable[Cell],
        ground_cells: Iterable[Cell],
        pred_cells: Iterable[Cell],
    ) -> str:
        bad_cells_str = tuples_to_str(bad_cells)

        ground_cells_str = tuples_to_str(ground_cells)

        pred_cells_str = tuples_to_str(pred_cells)

        if len(ground_cells_str) == 0 == len(pred_cells_str):
            if len(bad_cells_str) > 0:
                return "INVALID RELATION LABELS : {bad_cells_str}"
            return f"_no_{task_name}_errors_"
        bad_out = (
            f"INVALID RELATION LABELS : {bad_cells_str} , "
            if len(bad_cells_str) > 0
            else ""
        )
        return f"{bad_out}Ground: {ground_cells_str} , Predicted: {pred_cells_str}"

    def to_pred_string(reduced_matrix: np.ndarray) -> str:
        non_none_inds = np.where(reduced_matrix != none_index)
        non_none_cell_tuples = zip(
            *non_none_inds, reduced_matrix[non_none_inds].astype(int)
        )
        return tuples_to_str(non_none_cell_tuples)

    normalized_pred_pairs = (
        normalize_cells(pred, word_id_ls)
        for pred, word_id_ls in zip(resolved_predictions, word_ids)
    )
    if ground_truths is not None:
        normalized_ground_pairs = (
            normalize_cells(ground_truth, word_id_ls)
            for ground_truth, word_id_ls in zip(ground_truths, word_ids)
        )
        disagreements = (
            find_disagreements(ground_pair, pred_pair)
            for ground_pair, pred_pair in zip(
                normalized_ground_pairs, normalized_pred_pairs
            )
        )

        return [
            to_error_string(bad_cells, ground_cells, pred_cells)
            for bad_cells, ground_cells, pred_cells in disagreements
        ]
    return [
        to_pred_string(reduced_pred_matrix)
        for _, reduced_pred_matrix in normalized_pred_pairs
    ]
