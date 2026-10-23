# Test 1

def get(
    url: _t.UriType, params: _t.ParamsType = None, **kwargs: Unpack[_t.GetKwargs]
) -> Response:
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("get", url, params=params, **kwargs)

# Test 2

def get(
    url: _t.UriType, params: _t.ParamsType = None, **kwargs: Unpack[_t.GetKwargs]
) -> Response:
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("get", url, params=params, **kwargs)


def options(url: _t.UriType, **kwargs: Unpack[_t.RequestKwargs]) -> Response:

# Test 3

def options(url: _t.UriType, **kwargs: Unpack[_t.RequestKwargs]) -> Response:

# Test 4

def get(
    url: _t.UriType, params: _t.ParamsType = None, **kwargs: Unpack[_t.GetKwargs]
) -> Response:
    return request("get", url, params=params, **kwargs)