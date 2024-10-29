# visual_prompt_generator
generate visual prompt and reasoning dataset

## BBOX Maker 사용법

1. `python -m bbox_maker`

2. 사진 파일이 모여있는 폴더 선택

3. 커서를 이용해 bbox 생성

4. Backspace 로 이전 bbox redo 가능

5. Enter 입력시 파일 bbox 정보 저장

    같은 폴더에 `[사진이름]_bbox.npy` 로 저장 (List of tuples (x1, y1, x2, y2) normalized value [0, 1])

6. 좌우 방향키로 이전 이후 사진 리뷰 가능
