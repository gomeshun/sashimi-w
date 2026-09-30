"""Small immutable-input helpers; scientific settings are owned by this variant."""

import hashlib
import json

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.interpolate import interp1d


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def merge(defaults, previous, overrides):
    result = plain(defaults if previous is None else previous)
    for group, changes in overrides.items():
        if group not in result:
            raise TypeError(f"Unknown configuration group {group!r}.")
        if changes is None:
            continue
        if not isinstance(changes, Mapping):
            raise TypeError(f"{group} must be a mapping.")
        unknown = changes.keys() - result[group].keys()
        if unknown:
            raise ValueError(f"Unknown {group} options: {sorted(map(str, unknown))}.")
        for key, value in changes.items():
            if key == "solver_options":
                if not isinstance(value, Mapping):
                    raise TypeError("solver_options must be a mapping.")
                result[group][key] = {**result[group][key], **value} if value else {}
            else:
                result[group][key] = value
    return result


def scalar(value, name, minimum=0.0, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar.")
    result = float(raw)
    if not np.isfinite(result) or result < minimum or (positive and result == minimum):
        raise ValueError(
            f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}."
        )
    return result


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean.")
    return bool(value)


def choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}.")
    return value


def bounds(value, name, mass=False):
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a pair of bounds.")
    try:
        lower, upper = value
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain two bounds.") from error
    lower = scalar(lower, name + "[0]", positive=mass)
    if upper is None and mass:
        return lower, None
    upper = scalar(upper, name + "[1]", positive=mass)
    if upper <= lower:
        raise ValueError(f"{name} upper bound must exceed lower bound.")
    return lower, upper


def redshift_grid(target, requested, step, default_upper):
    explicit = requested is not None
    lo, hi = bounds(
        (target, default_upper) if requested is None else requested, "accretion_redshift_range"
    )
    if lo < target:
        raise ValueError("Accretion lower redshift must be at least the output redshift.")
    nodes = np.arange(lo + step, hi + step, step)
    if explicit:
        tolerance = 8 * np.finfo(float).eps * max(1.0, abs(hi))
        nodes = np.minimum(nodes[nodes <= hi + tolerance], hi)
    if nodes.size < 2:
        raise ValueError("Accretion support and redshift_step must yield at least two nodes.")
    nodes.setflags(write=False)
    return nodes, lo, hi, "explicit-bounded" if explicit else "legacy-default-grid"


def host_at_zero(model, mass, epoch):
    if epoch == 0.0:
        return mass
    candidates = np.logspace(0.0, 3.0, 1000) * mass
    values = np.asarray(model.Mzi(candidates, epoch))
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError("Host inversion must be finite and monotonic within its bracket.")
    if not values[0] <= mass <= values[-1]:
        raise ValueError("Host mass lies outside the supported inversion bracket.")
    return float(interp1d(values, candidates, bounds_error=True)(mass))


def ode_options(value, solver):
    if not isinstance(value, Mapping):
        raise TypeError("solver_options must be a mapping.")
    if solver != "odeint" and value:
        raise ValueError(
            "solver_options require solver='odeint'; clear incompatible options explicitly."
        )
    allowed = {"rtol", "atol", "h0", "hmax", "hmin", "mxstep", "mxhnil", "mxordn", "mxords"}
    if value.keys() - allowed:
        raise ValueError(f"Unsupported odeint options: {sorted(map(str, value.keys() - allowed))}.")
    result = {}
    for key, item in value.items():
        if key.startswith("mx"):
            result[key] = integer(item, key, 1 if key in {"mxordn", "mxords"} else 0)
            maximum = {"mxordn": 12, "mxords": 5}.get(key)
            if maximum is not None and result[key] > maximum:
                raise ValueError(f"{key} must not exceed {maximum}.")
        else:
            result[key] = scalar(
                item, key, minimum=-np.inf if key == "h0" else 0.0, positive=key in {"rtol", "atol"}
            )
    if result.get("h0", 0.0) > 0:
        raise ValueError("h0 must be nonpositive for decreasing-redshift integration.")
    if result.get("hmax", 0.0) > 0 and result.get("hmin", 0.0) > result["hmax"]:
        raise ValueError("hmin must not exceed a positive hmax.")
    return result


def settings_identifier(variant, settings):
    encoded = json.dumps(
        plain(settings), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return variant + ":native-settings:sha256:" + hashlib.sha256(encoded).hexdigest()
