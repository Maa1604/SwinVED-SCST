IMAGES_MIMIC_PATH = "/home/Data/NEW/mimic-cxr/2.0.0/files_jpg_512/files"

DICT_CSV_MIMIC_CXR_VQA_PATH = {
    "train": "../MIMIC-CXR-VQA/generated_questions_answers_train_all_small.csv",
    "validation": "../MIMIC-CXR-VQA/generated_questions_answers_validate_all.csv",
    "test": "../MIMIC-CXR-VQA/generated_questions_answers_test_all_small.csv"
}

VOCAB_PATH = "../MIMIC-CXR-VQA/vocab-mimic-cxr-vqa.tgt"

SWINB_IMAGENET22K_WEIGHTS = "microsoft/swin-base-patch4-window12-384-in22k"
SWINB_IMAGENET22K_WEIGHTS_FINETUNE = "../swin_mimic"