import mitsuba as mi

from nerad.texture import register_texture


@register_texture("dict")
class MiDictionary(mi.Texture):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        kwargs = {}
        for key in props.property_names():
            kwargs[key] = props.get(key)
        self.dict = kwargs
