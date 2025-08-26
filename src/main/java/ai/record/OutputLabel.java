package ai.record;

import ai.encoders.Encoder;


/**
 * Output value class being several different classes
 * @author lisztian
 *
 */
public interface OutputLabel<V> {

	public void setLabel(Object obj);
	public Object getLabel(V val);

	public void setEncoder(Encoder enoder);

	public V decode(Object obj);
}
