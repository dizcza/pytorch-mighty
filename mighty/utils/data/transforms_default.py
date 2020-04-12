from torchvision.transforms import Normalize, ToTensor, Compose, Resize


class TransformDefault:

    @staticmethod
    def mnist(normalize=Normalize(mean=(0.1307,), std=(0.3081,))):
        transforms = [ToTensor()]
        if normalize:
            transforms.append(normalize)
        return Compose(transforms)

    @staticmethod
    def cifar10():
        normalize = Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.247, 0.243, 0.261)
        )
        return Compose([ToTensor(), normalize])

    @staticmethod
    def imagenet():
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return Compose([Resize(size=(224, 224)), ToTensor(), normalize])
