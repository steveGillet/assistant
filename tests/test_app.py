from __future__ import annotations

from grapefruit.app import parse_args


def test_parse_defaults():
    args = parse_args([])
    assert args.wake_word == "grapefruit"
    assert args.voice == "eve"
    assert args.no_wake is False
    assert args.text_only is False


def test_parse_text_only_and_voice():
    args = parse_args(["--text-only", "--voice", "Ara", "--no-wake"])
    assert args.text_only is True
    assert args.voice == "Ara"
    assert args.no_wake is True


def test_parse_mic_and_barge_in():
    args = parse_args(["--mic-device", "3", "--barge-in"])
    assert args.mic_device == 3
    assert args.barge_in is True
    assert parse_args([]).barge_in is False
    assert parse_args([]).mic_device is None


def test_parse_idle_sec():
    assert parse_args(["--idle-sec", "120"]).idle_sec == 120
    assert parse_args([]).idle_sec is None
