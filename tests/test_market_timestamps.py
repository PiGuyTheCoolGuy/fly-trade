from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from flytrade.config import Config
from flytrade.market import Coinbase, DataError


class QuoteTimestampTests(unittest.TestCase):
    def setUp(self):
        self.client = Coinbase(Config())
        self.base = datetime(2026, 9, 14, 2, 26, 27, tzinfo=timezone.utc).timestamp()

    def quote_at(self, timestamp, now):
        raw = {"bids": [["100", "1", 1]], "asks": [["101", "1", 1]], "time": timestamp}
        with patch.object(self.client, "get", return_value=raw):
            return self.client.quote("BTC-USD", now)

    def test_reported_nanosecond_timestamps_are_accepted(self):
        examples = [
            ("02:26:27.592677893", 26, 27, 592677),
            ("02:26:42.654939461", 26, 42, 654939),
            ("02:26:58.118898748", 26, 58, 118898),
            ("02:27:12.792987132", 27, 12, 792987),
            ("02:27:27.798678953", 27, 27, 798678),
            ("02:27:43.293290421", 27, 43, 293290),
            ("02:27:57.782129991", 27, 57, 782129),
            ("02:28:12.903828224", 28, 12, 903828),
        ]
        for value, minute, second, microsecond in examples:
            expected = datetime(2026, 9, 14, 2, minute, second, microsecond, timezone.utc).timestamp()
            for suffix in ("Z", "+00:00"):
                with self.subTest(time=value, suffix=suffix):
                    result = self.quote_at(f"2026-09-14T{value}{suffix}", expected + 2)
                    self.assertEqual(result["timestamp"], expected)
                    self.assertEqual(result["mid"], 100.5)

    def test_whole_seconds_and_variable_fractional_precision(self):
        examples = [("", 0), (".1", 100000), (".12", 120000), (".123", 123000),
                    (".1234", 123400), (".12345", 123450), (".123456", 123456),
                    (".1234567", 123456), (".12345678", 123456), (".123456789", 123456)]
        for fraction, microsecond in examples:
            with self.subTest(fraction=fraction):
                expected = datetime(2026, 9, 14, 2, 26, 27, microsecond, timezone.utc).timestamp()
                result = self.quote_at(f"2026-09-14T02:26:27{fraction}Z", expected + 2)
                self.assertEqual(result["timestamp"], expected)

    def test_timezone_offsets_preserve_the_instant(self):
        for value in ("2026-09-14T04:56:27.592677893+02:30",
                      "2026-09-13T22:56:27.592677893-03:30"):
            with self.subTest(time=value):
                result = self.quote_at(value, self.base + 2)
                self.assertEqual(result["timestamp"], self.base + .592677)

    def test_nanosecond_quotes_still_obey_freshness_limits(self):
        timestamp = "2026-09-14T02:26:27.592677893Z"
        observed = self.base + .592677
        for age in (-6, 61):
            with self.subTest(age=age), self.assertRaisesRegex(DataError, "stale or future"):
                self.quote_at(timestamp, observed + age)

    def test_malformed_and_timezone_missing_timestamps_are_rejected(self):
        for value in (None, 123, {}, "", "not a timestamp", "2026-09-14",
                      "2026-09-14T02:26:27", "2026-09-14T02:26:27.592677893",
                      "2026-09-14T02:26:27.Z", "2026-09-14T02:26:27.592oopsZ",
                      "2026-13-14T02:26:27Z", "2026-09-14T02:26:27Zjunk"):
            with self.subTest(time=value), self.assertRaises(DataError):
                self.quote_at(value, self.base + 2)


if __name__ == "__main__":
    unittest.main()
