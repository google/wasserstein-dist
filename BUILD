py_binary(
    name = "compute_one",
    srcs = ["compute_one.py"],
    deps = [
        ":dataset",
        ":wasserstein",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_binary(
    name = "compute_all",
    srcs = ["compute_all.py"],
    deps = [
        ":dataset",
        ":wasserstein",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        ":generator",
        ":wasserstein",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "wasserstein",
    srcs = ["wasserstein.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/tensorflow/contrib/slim",
    ],
)
