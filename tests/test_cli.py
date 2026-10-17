import csv

from oddmeter.cli import main


def test_demo_command_writes_report(tmp_path, monkeypatch, capsys):
    out = tmp_path / "nested" / "report.csv"
    monkeypatch.setattr("sys.argv", ["oddmeter", "demo", "--output", str(out)])
    main()
    assert out.exists()
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 600
    assert {"is_anomaly", "anomaly_score", "main_signal", "known_event"} <= set(rows[0])
    assert "precision" in capsys.readouterr().out


def test_analyze_csv(tmp_path, monkeypatch, capsys):
    path = tmp_path / "points.csv"
    out = tmp_path / "report.csv"
    path.write_text("id,x,y\n" + "".join(f"{i},{i},{i*2}\n" for i in range(40)) + "999,5000,-4000\n")
    monkeypatch.setattr("sys.argv", ["oddmeter", "analyze", str(path), "--output", str(out), "--contamination", "0.05"])
    main()
    assert out.exists()
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 41
    assert rows[-1]["is_anomaly"] == "True"
    assert "Analyzed 41 rows" in capsys.readouterr().out
