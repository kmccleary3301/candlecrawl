from pathlib import Path

REQUIRED_SNIPPETS = [
    "openapi:",
    "/v2/scrape",
    "/v2/map",
    "/v2/crawl",
]


def main() -> int:
    contract_path = Path("contracts/openapi-v1.yaml")
    if not contract_path.exists():
        raise SystemExit("Contract check failed: contracts/openapi-v1.yaml is missing")

    text = contract_path.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Contract check failed: missing required entries: {missing_list}")

    print("Contract check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
