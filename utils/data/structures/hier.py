import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Hier(object):
    def __init__(self, hier, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = hier.device if isinstance(hier, torch.Tensor) else torch.device('cpu')
        hier = torch.as_tensor(hier, dtype=torch.float32, device=device)
        num_hier = hier.shape[0]
        if num_hier:
            hier = hier.view(num_hier, -1, 5)

        # TODO should I split them?
        # self.visibility = hier[..., 4]
        self.hier = hier  # [..., :4]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.hier.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        resized_data[..., 2] *= ratio_w
        resized_data[..., 3] *= ratio_h
        hier = Hier(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            hier.add_field(k, v)
        return hier

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = torch.as_tensor([0, 1, 3, 2, 5, 4])
        flipped_data = self.hier[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
        flipped_data[..., 2] = width - flipped_data[..., 0] - TO_REMOVE

        hier = Hier(self.hier, self.size, self.mode)
        for k, v in self.extra_fields.items():
            hier.add_field(k, v)
        return hier

    def to(self, *args, **kwargs):
        hier = Hier(self.hier.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            hier.add_field(k, v)
        return hier

    def __getitem__(self, item):
        hier = Hier(self.hier[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            hier.add_field(k, v[item])
        return hier

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.hier))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s
