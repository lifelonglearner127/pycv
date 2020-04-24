class Error(Exception):
    pass


class ViolaJonesFaceDetectorConfError(Error):
    pass


class DLibCNNFaceDetectorConfError(Error):
    pass


class CascadeFileDoesNotExist(ViolaJonesFaceDetectorConfError):
    pass


class CaffeDNNFaceDetectorConfError(Error):
    pass


class CaffeModelFileDoesNotExist(CaffeDNNFaceDetectorConfError):
    pass


class CaffeConfigFileDoesNotExist(CaffeDNNFaceDetectorConfError):
    pass


class CaffeModelConfigurationError(CaffeDNNFaceDetectorConfError):
    pass
