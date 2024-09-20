SELECT stat.delivery_order_id,
       stat.order_date,
       stat.order_id,
       stat.distance,
       stat.price_gross_per_ton,
       stat.price_net_per_ton,
       stat.disassembled_weight,
       stat.hours_until_disassembled,
       stat.region_code,
       stat.status,
       stat.ctime,
       crops_id,
       dest_title
FROM (SELECT DISTINCT ON (order_id ) order_date,
                                     order_id,
                                     delivery_order_id,
                                     distance,
                                     price_gross_per_ton,
                                     price_net_per_ton,
                                     disassembled_weight,
                                     hours_until_disassembled,
                                     region_code,
                                     status,
                                     ctime
      FROM delivery_orders_stat
      WHERE order_date >= '@date_from'
      ORDER BY order_id, ctime, delivery_order_id) AS stat
         INNER JOIN ((SELECT DISTINCT ON (delivery_order_id) order_id, delivery_order_id, crops_id
                      FROM delivery_orders
                      ORDER BY delivery_order_id) as dorders INNER JOIN (SELECT DISTINCT ON (order_id) order_id, dest_title
                                                                         FROM orders
                                                                         WHERE order_date >= '@date_from'
                                                                         ORDER BY order_id) as sorders
                     ON dorders.order_id = sorders.order_id) as t ON stat.delivery_order_id = t.delivery_order_id
