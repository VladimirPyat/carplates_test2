# images_apply

- moved: 66
- fixed: 15
- skipped: 0
- quarantine: /work/plates_detect_test/dataset_prepare/audit_out/stage1_audit_files/quarantine

## Осознанно пропущено

- Геометрия боксов, дубли боксов, min размер в px, маппинг классов — этап 2
- Похожие (не точные) дубли — этап 3
- Blur / экспозиция / цвет — пропущено осознанно, просмотр в CVAT
- EXIF: только флаг, нормализация не выполнялась
- RGBA: альфа отбрасывается без композита по допущению , перепроверить позже
- Сплит / стратификация / data.yaml — этап 3
