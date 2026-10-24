"""host_fingerprint returns real machine values (stdlib probes only)."""

from __future__ import annotations

import platform

from runner_protocol import Host, host_fingerprint


def test_fingerprint_shape_and_values() -> None:
    fingerprint = host_fingerprint()
    assert set(fingerprint) == {"hostname", "cpu", "ram_gb", "gpus", "os"}
    assert fingerprint["hostname"] == platform.node()
    assert isinstance(fingerprint["cpu"], str) and fingerprint["cpu"]
    assert isinstance(fingerprint["ram_gb"], float) and fingerprint["ram_gb"] > 0.0
    assert isinstance(fingerprint["gpus"], list)
    for gpu in fingerprint["gpus"]:
        assert set(gpu) == {"model", "vram_gb"}
    assert isinstance(fingerprint["os"], str) and fingerprint["os"]


def test_fingerprint_validates_against_wire_model() -> None:
    host = Host.model_validate(host_fingerprint())
    assert host.hostname == platform.node()
    assert host.ram_gb > 0.0
