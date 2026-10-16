def get(
    url: _t.UriType, params: _t.ParamsType = None, **kwargs: Unpack[_t.GetKwargs]
) -> Response:
    return request("get", url, params=params, **kwargs)