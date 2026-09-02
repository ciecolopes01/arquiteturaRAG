import sys; sys.path.insert(0,'.')
from src.config import settings
from qdrant_client import QdrantClient
client = QdrantClient(url=settings.qdrant_url)
ids = [
    '730440d1-8c15-44d2-a14a-b1efb14361af','1d3054be-d70c-4e18-ae6b-bfffd35fddd7',
    'a181ac61-6cdd-44a1-ab03-fda707413c53','a24a5f52-e02c-4715-8804-a548c23c95d4',
    '45bd280b-9a94-4ece-ba38-0bffd23c5797','1fa5bf5e-5d70-4484-b175-94029f69d7c2',
    '9bc2abdd-7f54-41c4-8738-8596fbd01000','2201a399-f872-4f66-8943-83fed1881b36',
    '2f0bb7ac-42ad-4a65-880b-25613e25bbe4','c2635cf1-acf0-495d-a335-943559edf02a',
    '37a12271-f097-4198-94a1-af59b99d51a7','dd69e77c-ce4b-4734-8565-93d0ea779ef6',
    '967e525f-6d0a-48cf-8c27-550e2e4deacc','4b9304a2-bbe6-4c58-b68b-00da90922c96',
    '6d29c77e-714c-4e0a-8aa0-ac448662a377','c38500dc-d3cc-4153-8cc5-3ac173db9eea',
]
r = client.retrieve(collection_name=settings.qdrant_collection, ids=ids, with_payload=True)
for p in sorted(r, key=lambda x: (x.payload.get('source_uri',''), x.payload.get('chunk_index',0))):
    src = p.payload.get('source_uri','').split('\\')[-1]
    idx = p.payload.get('chunk_index','?')
    sec = (p.payload.get('section_title') or '')[:40]
    content = p.payload.get('content','')[:120].replace('\n',' ')
    print(f"[{src}][{idx}] {p.id}")
    print(f"  sec={sec}")
    print(f"  '{content}'")
    print()
