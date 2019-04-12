import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    Function to calculate the FID score
    :param sample_feature_iterator:
    :param testset_feature_iterator:
    :return:
    """
    mu_q = torch.zeros(512)
    sm_q = torch.zeros(512, 512)
    nb_batches = 0
    for i, feature in enumerate(sample_feature_iterator):
        feature = torch.from_numpy(feature.astype('float32'))
        nb_batches = i + 1
        mu_q += feature
        sm_q += torch.matmul(feature.view(512, 1), feature.view(1, 512))

    # get the first and second moment estimates of the distribution
    mu_q /= nb_batches
    sm_q /= nb_batches

    # get sigma for q
    sigma_q = sm_q - torch.matmul(mu_q.view(512, 1), mu_q.view(1, 512))

    # do the same for p
    mu_p = torch.zeros_like(mu_q)
    sm_p = torch.zeros_like(sm_q)
    nb_batches = 0
    for i, feature in enumerate(testset_feature_iterator):
        feature = torch.from_numpy(feature.astype('float32'))
        nb_batches = i + 1
        mu_p += feature
        sm_p += torch.matmul(feature.view(512, 1), feature.view(1, 512))

    mu_p /= nb_batches
    sm_p /= nb_batches

    sigma_p = sm_p - torch.matmul(mu_p.view(512, 1), mu_p.view(1, 512))

    # compute the FID score
    fid = torch.norm(mu_q - mu_p) ** 2. + torch.trace(sigma_q + sigma_p -2 * torch.matmul(sigma_p, sigma_q) ** 2.)

    return fid.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
